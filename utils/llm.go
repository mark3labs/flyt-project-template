package utils

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// LLMConfig holds configuration for LLM calls
type LLMConfig struct {
	Model       string  `json:"model"`
	Temperature float64 `json:"temperature"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
}

// DefaultLLMConfig returns default configuration
func DefaultLLMConfig() *LLMConfig {
	return &LLMConfig{
		Model:       "gpt-3.5-turbo",
		Temperature: 0.7,
		MaxTokens:   0, // Use model default
	}
}

// CallLLM calls the OpenAI API with the given prompt
func CallLLM(prompt string) (string, error) {
	return CallLLMWithConfig(prompt, DefaultLLMConfig())
}

// CallLLMWithConfig calls the OpenAI API with custom configuration
func CallLLMWithConfig(prompt string, config *LLMConfig) (string, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return "", fmt.Errorf("OPENAI_API_KEY environment variable not set")
	}

	// Prepare request body
	requestBody := map[string]any{
		"model": config.Model,
		"messages": []map[string]string{
			{
				"role":    "system",
				"content": "You are a helpful assistant.",
			},
			{
				"role":    "user",
				"content": prompt,
			},
		},
		"temperature": config.Temperature,
	}

	if config.MaxTokens > 0 {
		requestBody["max_tokens"] = config.MaxTokens
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	req, err := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	// Make request with timeout
	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	// Read response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}

	if len(result.Choices) == 0 {
		return "", fmt.Errorf("no response from API")
	}

	return result.Choices[0].Message.Content, nil
}

// CallLLMStreaming calls the OpenAI API with streaming response
// This is useful for long responses where you want to show progress
func CallLLMStreaming(prompt string, onChunk func(string) error) error {
	// Implementation would handle streaming responses
	// For now, we'll use the regular call
	response, err := CallLLM(prompt)
	if err != nil {
		return err
	}

	return onChunk(response)
}
