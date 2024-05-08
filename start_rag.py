from rag.query_to_gpt import query_to_gpt

def main():
    print("Remember that right now our application database has only 200 papers, so the query should be related to those papers.")
    print("Please enter a query related to the following topics: ")
    print("- Singularity Studies in Manifolds")
    print("- Geodesic Surfaces in String Theory")
    print("- Particle Physics")
    print("- Quantum Information Processing")
    print("- Anisotropic Magnetic Properties")

    user_query = input("Enter your query: ")
    result = query_to_gpt(query=user_query)
    print(result)

if __name__ == "__main__":
    main()
