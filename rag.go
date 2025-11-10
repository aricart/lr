package main

import (
	"fmt"
	"strings"
)

// RAG handles retrieval-augmented generation
type RAG struct {
	VectorStore      *VectorStore
	MultiSourceStore *MultiSourceStore
	LLM              LLMClient
}

// NewRAG creates a new RAG system with a single vector store
func NewRAG(vs *VectorStore, llm LLMClient) *RAG {
	return &RAG{
		VectorStore: vs,
		LLM:         llm,
	}
}

// NewRAGMultiSource creates a new RAG system with multi-source support
func NewRAGMultiSource(mss *MultiSourceStore, llm LLMClient) *RAG {
	return &RAG{
		MultiSourceStore: mss,
		LLM:              llm,
	}
}

// Query performs a RAG query across all sources
func (r *RAG) Query(question string, topK int) (string, []SearchResult, error) {
	return r.QueryWithSources(question, topK, []string{})
}

// QueryWithSources performs a RAG query on specific sources
func (r *RAG) QueryWithSources(question string, topK int, sources []string) (string, []SearchResult, error) {
	// get embedding for the question
	queryEmbedding, err := r.LLM.GetEmbedding(question)
	if err != nil {
		return "", nil, fmt.Errorf("failed to get query embedding: %w", err)
	}

	// search for relevant chunks (use multi-source if available)
	var results []SearchResult
	if r.MultiSourceStore != nil {
		results = r.MultiSourceStore.Search(queryEmbedding, topK, sources)
	} else {
		results = r.VectorStore.Search(queryEmbedding, topK)
	}

	// build context from top results
	var contextBuilder strings.Builder
	contextBuilder.WriteString("here is the relevant context from the indexed documentation and source code:\n\n")

	for i, result := range results {
		contextBuilder.WriteString(fmt.Sprintf("--- document %d (source: %s, type: %s, similarity: %.3f) ---\n",
			i+1, result.Chunk.Source, result.Chunk.Metadata["type"], result.Similarity))
		contextBuilder.WriteString(result.Chunk.Text)
		contextBuilder.WriteString("\n\n")
	}

	// build prompt
	systemPrompt := `you are a helpful assistant that answers questions based on indexed documentation and source code.
answer based solely on the provided context from the indexed repositories.
if the context doesn't contain enough information to answer the question, say so.
always cite the source documents when answering.
when showing code examples, preserve the formatting and explain what the code does.`

	userPrompt := fmt.Sprintf("%s\n\nquestion: %s", contextBuilder.String(), question)

	messages := []Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	// get response from llm
	answer, err := r.LLM.Chat(messages)
	if err != nil {
		return "", results, fmt.Errorf("failed to get chat response: %w", err)
	}

	return answer, results, nil
}
