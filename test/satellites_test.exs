defmodule SolverTest do
  use ExUnit.Case
  doctest Solver

  test "solve" do
    assert Solver.solve == ["START", "SAT3", "SAT17", "END"]
  end
end
