from graph import Graph
from graph.shapes import Point_Cloud
d = Graph("http://127.0.0.1:5000/")

print(Point_Cloud.from_uuid("http://127.0.0.1:5000/", "464a013f-150e-4128-afd6-e82748b74fb7"))
