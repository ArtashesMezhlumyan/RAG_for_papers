from rag.query_to_gpt import query_to_gpt


# remember that right now our application database has only 200 papers, so the query should be related to those papers
# Suggested topics -  Singularity Studies in Manifolds, Geodesic Surfaces in String Theory, Particle PhysicsQuantum Information Processing, Anisotropic Magnetic Properties
print(query_to_gpt(query = "Your query here"))
