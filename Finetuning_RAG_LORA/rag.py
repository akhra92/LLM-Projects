import torch
import torch.nn as nn
from typing import List, Tuple


class DocumentRetriever:    
    def __init__(self, embedding_dim=768):
        self.embedding_dim = embedding_dim
        # Knowledge base: store document embeddings
        self.document_embeddings = []
        self.documents = []

    def add_documents(self, documents: List[str], embeddings: torch.Tensor):        
        self.documents.extend(documents)
        self.document_embeddings.append(embeddings)

    def retrieve(self, query_embedding: torch.Tensor, top_k: int = 3) -> List[Tuple[str, float]]:
        if not self.document_embeddings:
            return []

        # Concatenate all document embeddings
        all_embeddings = torch.cat(self.document_embeddings, dim=0)

        # Compute cosine similarity: dot product of normalized vectors
        query_norm = query_embedding / query_embedding.norm()
        doc_norms = all_embeddings / all_embeddings.norm(dim=1, keepdim=True)
        similarities = torch.matmul(doc_norms, query_norm)

        # Get top-k most similar documents
        top_k = min(top_k, len(self.documents))
        top_scores, top_indices = torch.topk(similarities, top_k)

        results = [(self.documents[idx], score.item())
                   for idx, score in zip(top_indices, top_scores)]
        return results


class RAGModel(nn.Module):    
    def __init__(self, embedding_dim=768, hidden_dim=512, vocab_size=50000):
        super().__init__()
        # Document retriever
        self.retriever = DocumentRetriever(embedding_dim)

        # Query encoder (in practice, use pre-trained model like BERT)
        self.query_encoder = nn.Linear(vocab_size, embedding_dim)

        # Generator (in practice, use pre-trained LM like GPT, T5)
        self.generator = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),  # query + context
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def encode_query(self, query: torch.Tensor) -> torch.Tensor:
        """Encode query into embedding space"""
        return self.query_encoder(query)

    def forward(self, query: torch.Tensor, top_k: int = 3) -> Tuple[torch.Tensor, List[str]]:
        # Step 1: Encode query
        query_embedding = self.encode_query(query)

        # Step 2: Retrieve relevant documents
        retrieved = self.retriever.retrieve(query_embedding[0], top_k)
        retrieved_docs = [doc for doc, _ in retrieved]

        # Step 3: Combine query with retrieved context (Simulate)
        context_embedding = torch.randn_like(query_embedding)
        combined = torch.cat([query_embedding, context_embedding], dim=-1)

        # Step 4: Generate response
        output = self.generator(combined)

        return output, retrieved_docs


if __name__ == "__main__":
    print("=== RAG Demo ===\n")

    # Initialize model
    embedding_dim = 768
    vocab_size = 50000
    model = RAGModel(embedding_dim=embedding_dim, vocab_size=vocab_size)

    # Add documents to knowledge base
    documents = [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neurons in the brain.",
        "RAG combines retrieval and generation for better LLM outputs."
    ]
    # Simulate document embeddings (in practice, use BERT/Sentence-BERT)
    doc_embeddings = torch.randn(len(documents), embedding_dim)
    model.retriever.add_documents(documents, doc_embeddings)

    # Process query
    query = torch.randn(1, vocab_size)  # Simulate encoded query
    output, retrieved_docs = model(query, top_k=2)

    print(f"Query shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nRetrieved documents:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"{i}. {doc}")

    # Calculate model memory
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
