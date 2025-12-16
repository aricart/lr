package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// OllamaClient handles Ollama local API requests for embeddings
type OllamaClient struct {
	BaseURL string
	Model   string
	Client  *http.Client
}

// NewOllamaClient creates a new Ollama client
func NewOllamaClient(model string) *OllamaClient {
	if model == "" {
		model = "nomic-embed-text"
	}
	return &OllamaClient{
		BaseURL: "http://localhost:11434",
		Model:   model,
		Client:  &http.Client{Timeout: 30 * time.Second},
	}
}

// OllamaEmbedRequest represents an Ollama embedding request
type OllamaEmbedRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

// OllamaEmbedResponse represents an Ollama embedding response
type OllamaEmbedResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
}

// GetEmbedding gets an embedding for the given text using Ollama
func (o *OllamaClient) GetEmbedding(text string) ([]float64, error) {
	reqBody := OllamaEmbedRequest{
		Model: o.Model,
		Input: text,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", o.BaseURL+"/api/embed", bytes.NewBuffer(body))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := o.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama not running? %w (start with: ollama serve)", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama error: %s - %s", resp.Status, string(bodyBytes))
	}

	var embResp OllamaEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&embResp); err != nil {
		return nil, err
	}

	if len(embResp.Embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned from ollama")
	}

	return embResp.Embeddings[0], nil
}

// Chat is not supported by Ollama embeddings client
func (o *OllamaClient) Chat(_ []Message) (string, error) {
	return "", fmt.Errorf("ollama embeddings client does not support chat - use with claude")
}

// OllamaClaudeClient uses Ollama for embeddings and Claude for chat
type OllamaClaudeClient struct {
	Ollama *OllamaClient
	Claude *AnthropicClient
}

// NewOllamaClaudeClient creates a client using Ollama embeddings + Claude chat
// Returns an error if ANTHROPIC_API_KEY is not set
func NewOllamaClaudeClient(embeddingModel, chatModel string) (*OllamaClaudeClient, error) {
	claudeKey := os.Getenv("ANTHROPIC_API_KEY")
	if claudeKey == "" {
		return nil, fmt.Errorf("ANTHROPIC_API_KEY is required for ollama+claude mode")
	}
	return &OllamaClaudeClient{
		Ollama: NewOllamaClient(embeddingModel),
		Claude: NewAnthropicClient(claudeKey, chatModel),
	}, nil
}

// GetEmbedding uses Ollama for embeddings
func (oc *OllamaClaudeClient) GetEmbedding(text string) ([]float64, error) {
	return oc.Ollama.GetEmbedding(text)
}

// Chat uses Claude for chat
func (oc *OllamaClaudeClient) Chat(messages []Message) (string, error) {
	return oc.Claude.Chat(messages)
}
