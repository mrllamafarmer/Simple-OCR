// src/App.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [files, setFiles] = useState([]);
  const [provider, setProvider] = useState('OpenAI');
  const [model, setModel] = useState('');
  const [models, setModels] = useState([]);
  const [outputFormat, setOutputFormat] = useState('json');
  const [loading, setLoading] = useState(false);

  // Use environment variables for API URLs
  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8300';

  useEffect(() => {
    fetchModels(provider);
  }, [provider]);

  const fetchModels = async (selectedProvider) => {
    try {
      const response = await axios.get(`${apiUrl}/models/${selectedProvider}`);
      setModels(response.data);
      setModel(response.data[0]); // Set the first model as default
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const handleFileChange = (e) => {
    setFiles(Array.from(e.target.files));
  };

  const handleProviderChange = (e) => {
    setProvider(e.target.value);
    setModel(''); // Reset model when provider changes
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));
    formData.append('provider', provider);
    formData.append('model', model);
    formData.append('output_format', outputFormat);

    try {
      const response = await axios.post(`${apiUrl}/ocr`, formData, {
        responseType: 'blob',
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `output.${outputFormat}`);
      document.body.appendChild(link);
      link.click();
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>OCR Application</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} accept=".png,.jpg,.jpeg,.pdf" multiple required />
        <select value={provider} onChange={handleProviderChange}>
          <option value="OpenAI">OpenAI</option>
          <option value="OpenRouter">OpenRouter</option>
        </select>
        <select value={model} onChange={(e) => setModel(e.target.value)} required>
          <option value="">Select a model</option>
          {models.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
        <select value={outputFormat} onChange={(e) => setOutputFormat(e.target.value)}>
          <option value="json">JSON</option>
          <option value="txt">TXT</option>
        </select>
        <button type="submit" disabled={files.length === 0 || !model || loading}>
          {loading ? 'Processing...' : 'Submit'}
        </button>
      </form>
    </div>
  );
}

export default App;