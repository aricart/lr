package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// VoyageClient handles Voyage AI API requests
type VoyageClient struct {
	APIKey string
	Client *http.Client
}

// NewVoyageClient creates a new Voyage AI client
func NewVoyageClient(apiKey string) *VoyageClient {
	return &VoyageClient{
		APIKey: apiKey,
		Client: &http.Client{},
	}
}

// VoyageEmbeddingRequest represents a Voyage embedding request
type VoyageEmbeddingRequest struct {
	Input []string `json:"input"`
	Model string   `json:"model"`
}

// VoyageEmbeddingResponse represents a Voyage embedding response
type VoyageEmbeddingResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
}

// GetEmbedding gets an embedding for the given text using Voyage AI
func (v *VoyageClient) GetEmbedding(text string) ([]float64, error) {
	reqBody := VoyageEmbeddingRequest{
		Input: []string{text},
		Model: "voyage-code-2", // optimized for code
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", "https://api.voyageai.com/v1/embeddings", bytes.NewBuffer(body))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+v.APIKey)

	resp, err := v.Client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("voyage ai error: %s - %s", resp.Status, string(bodyBytes))
	}

	var embResp VoyageEmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&embResp); err != nil {
		return nil, err
	}

	if len(embResp.Data) == 0 {
		return nil, fmt.Errorf("no embeddings returned from voyage ai")
	}

	return embResp.Data[0].Embedding, nil
}

// Chat is not supported by Voyage (they only do embeddings)
func (v *VoyageClient) Chat(messages []Message) (string, error) {
	return "", fmt.Errorf("voyage ai does not support chat - use claude or openai")
}

// VoyageClaudeClient uses Voyage for embeddings and Claude for chat
type VoyageClaudeClient struct {
	Voyage *VoyageClient
	Claude *AnthropicClient
}

// NewVoyageClaudeClient creates a client using Voyage embeddings + Claude chat
func NewVoyageClaudeClient(voyageKey, claudeKey string) *VoyageClaudeClient {
	return &VoyageClaudeClient{
		Voyage: NewVoyageClient(voyageKey),
		Claude: NewAnthropicClient(claudeKey),
	}
}

// GetEmbedding uses Voyage for embeddings
func (vc *VoyageClaudeClient) GetEmbedding(text string) ([]float64, error) {
	return vc.Voyage.GetEmbedding(text)
}

// Chat uses Claude for chat
func (vc *VoyageClaudeClient) Chat(messages []Message) (string, error) {
	return vc.Claude.Chat(messages)
}
