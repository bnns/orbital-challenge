defmodule Solver do

  @moduledoc """
  Solves for the shortest path for routing phone calls through a network of satellites - see https://reaktor.com/orbital-challenge/
  """
  @header [:name, :lat, :long, :alt]
  @radius 6371
  @seed 0.37062555085867643
  @start_coords %{lat: "89.0299545467428", long: "-159.91518633147066"}
  @end_coords %{lat: "-25.726476388970326", long: "-25.726476388970326"}
  @start_name "START"
  @end_name "END"

  defp neighbor_distance_from(node) do
    fn neighbor ->
      dist = calc_distance node, neighbor
      set_unvisited neighbor, dist
    end
  end

  defp euclidean(start, finish) do
    # euclidean distance
    %{x: x1, y: y1, z: z1} = start
    %{x: x2, y: y2, z: z2} = finish
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    dist_squared = :math.pow(dx, 2) + :math.pow(dy, 2) + :math.pow(dz, 2)
    :math.sqrt dist_squared
  end

  defp calc_distance(start, finish) do

    dist = euclidean(start, finish)

    cond do
      start[:distance] == :infinity ->
        dist
      finish[:distance] == :infinity ->
        start[:distance] + dist
      start[:distance] + dist < finish[:distance] ->
        start[:distance] + dist
      true ->
        finish[:distance]
    end
  end

  defp to_distance(node) do
    node[:distance]
  end

  defp to_name(node) do
    node[:name]
  end

  defp get_node_values(graph) do
    fn name ->
      Enum.find(graph, fn v -> v[:name] == name end)
    end
  end

  defp not_visited(node) do
    !node[:visited]
  end

  defp traverse_unvisited(name, q) do
    # for each node, find distance of all neighbors and assign nearest
    # neighbor as the next current node

    get_values_from_name = get_node_values(q)
    node = get_values_from_name.(name)
    neighbor_distance = neighbor_distance_from(node)

    new_neighbors = node[:neighbors]
    |> Enum.map(&(get_values_from_name.(&1)))
    |> Enum.filter(&not_visited(&1))
    |> Enum.map(&(neighbor_distance.(&1)))

    new_q = update_q(node, q, new_neighbors)

    case Enum.empty?(new_neighbors) do
      true -> new_q
      false ->
        nearest_neighbor = new_neighbors
        |> Enum.min_by(&to_distance(&1))
        traverse_unvisited(nearest_neighbor[:name], new_q)
    end
  end

  defp not_in_list(list, key) do
    fn (item) ->
      # check if the name exists in list
      Enum.all?(list, fn v -> item[key] != v[key] end)
    end
  end

  defp update_q(node, old_list, new_list) do
    # update the old list with the new neighbors
    visited_node = set_unvisited(node, true)

    old_list
    |> Enum.filter(not_in_list(new_list ++ [node], :name))
    |> Enum.concat(new_list ++ [visited_node])
  end

  defp set_unvisited(v, val) when is_float(val) do
    %{v | :distance => val}
  end

  defp set_unvisited(v, val) when is_boolean(val) do
    %{v | :visited => val}
  end

  defp set_unvisited(%{name: "START"} = v) do
    Map.merge(v, %{distance: 0.0, visited: false})
  end

  defp set_unvisited(v) do
    Map.merge(v, %{distance: :infinity, visited: false})
  end

  defp find_sequence(get, target) do
    prev = target[:neighbors]
    |> Enum.map(get)
    |> Enum.min_by(&to_distance(&1))

    case prev do
      %{name: "START"} -> [prev] ++ [target]
      _ -> find_sequence(get, prev) ++ [target]
    end
  end

  @doc """
  Find the shortest path between two vertices in a graph
  https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
  """
  def dijkstra(graph) do
    q = graph |> Enum.map(&set_unvisited(&1))

    q_with_distances = [@start_name]
    |> Enum.reduce(q, &traverse_unvisited(&1, &2))

    get_values_from_name = get_node_values(q_with_distances)
    finish = get_values_from_name.(@end_name)

    find_sequence(&(get_values_from_name.(&1)), finish)
    |> IO.inspect
    |> Enum.map(&to_name(&1))
  end

  def solve do
    start = Map.merge(%{name: @start_name, alt: "100.0"}, @start_coords)
    finish = Map.merge(%{name: @end_name, alt: "100.0"}, @end_coords)
    satellites = File.stream!("input.csv")
    |> CSV.decode
    |> Enum.map(&toSatellite(&1))

    vertices = ([start] ++ satellites ++ [finish])
    |> Enum.map(&convertCoordinates(&1))

    graph = vertices
    |> Enum.map(&findNeighbors(&1, vertices))

    dijkstra(graph)
  end

  @doc """
  Converts lat, long and alt to x, y, z
  http://gis.stackexchange.com/questions/4147/lat-lon-alt-to-spherical-or-cartesian-coordinates
  https://en.wikipedia.org/wiki/Geographic_coordinate_conversion
  """
  def convertCoordinates(point) do
    phi = String.to_float(point[:lat]) * :math.pi / 180
    theta = String.to_float(point[:long]) * :math.pi / 180
    rho = String.to_float(point[:alt]) + @radius
    x = :math.sin(phi) * :math.cos(theta) * rho
    y = :math.sin(phi) * :math.sin(theta) * rho
    z = :math.cos(phi) * rho
    Map.merge(point, %{x: x, y: y, z: z})
  end

  defp toSatellite(row) do
    row
    |> (&Enum.zip(@header, &1)).()
    |> Enum.into(%{})
  end

  defp findNeighbors(sat, list) do
    neighbors = list
    |> Enum.filter(&lineBetween(&1, sat))
    |> Enum.map(&to_name(&1))

    Map.put_new(sat, :neighbors, neighbors)
  end

  @doc """
  Given cartesian coordinates of two points and the radius of the Earth, find if there exists a clear line of sight
  http://paulbourke.net/geometry/circlesphere/index.html#linesphere
  """
  def lineBetween(initial, final) do
    identical = initial[:name] == final[:name]

    %{:x => x0, :y => y0, :z => z0} = initial
    %{:x => x1, :y => y1, :z => z1} = final

    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0

    a = :math.pow(dx, 2) +
    :math.pow(dy, 2) +
    :math.pow(dz, 2)
    b = 2 * dx * x0 +
    2 * dy * y0 +
    2 * dz * z0
    c = :math.pow(x0, 2) +
    :math.pow(y0, 2) +
    :math.pow(z0, 2) -
    :math.pow(@radius, 2)

    (b * b - 4 * a * c <= 0) && !identical
  end

end
