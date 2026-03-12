import { useState, useRef, ChangeEvent, DragEvent } from 'react'
import { FileText, UploadCloud, File, AlertCircle, CheckCircle2, ChevronRight, BarChart2, Calendar, MapPin, DollarSign, Building2, Loader2 } from 'lucide-react'
import './index.css'

interface Metrics {
  text_length: number;
  word_count: number;
  noise_ratio: number;
  alpha_ratio: number;
}

interface Entities {
  DATE: string[];
  PARTY: string[];
  AMOUNT: string[];
  JURISDICTION: string[];
}

interface APIResponse {
  document_id: string;
  filename: string;
  metrics: Metrics;
  entities: Entities;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isExtracting, setIsExtracting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<APIResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type === 'application/pdf') {
        setFile(droppedFile);
        setError(null);
      } else {
        setError("Only PDF files are supported.");
      }
    }
  };

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleExtract = async () => {
    if (!file) return;
    
    setIsExtracting(true);
    setProgress(15);
    setError(null);
    setResult(null);
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Simulate initial progress while uploading
    const progressInterval = setInterval(() => {
      setProgress(p => p < 85 ? p + 2 : p);
    }, 200);

    try {
      const response = await fetch('http://localhost:8000/extract', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setProgress(100);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to extract entities");
      }

      const data: APIResponse = await response.json();
      
      // Artificial delay to make the 100% progress visible and feel smooth
      setTimeout(() => {
        setResult(data);
        setIsExtracting(false);
      }, 500);
      
    } catch (err: any) {
      clearInterval(progressInterval);
      setError(err.message || "An unexpected error occurred.");
      setIsExtracting(false);
      setProgress(0);
    }
  };

  const resetForm = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setProgress(0);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* Navigation */}
      <nav className="navbar">
        <div className="container flex items-center justify-between">
          <div className="brand">
            <span className="brand-icon"><FileText size={28} /></span>
            <span>LexiScan<span className="text-secondary opacity-70">Auto</span></span>
          </div>
          <div>
            <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" className="btn btn-secondary text-sm px-4 py-2">
              API Docs
            </a>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="container flex-col items-center gap-8 py-12" style={{ flexGrow: 1 }}>
        
        {/* Header Section */}
        {!result && (
          <div className="flex-col items-center text-center max-w-2xl mx-auto mb-12 animate-fade-in">
            <h1 className="text-4xl md:text-5xl font-extrabold mb-4">
              Extract Legal Entities <br/>
              <span className="text-gradient">in Seconds.</span>
            </h1>
            <p className="text-subtitle font-medium">
              Upload scanned or native PDF contracts. LexiScan automatically extracts parties, dates, amounts, and jurisdictions into normalized, structured data.
            </p>
          </div>
        )}

        {/* Workspace Area */}
        <div className="w-full max-w-4xl mx-auto">
          
          {/* Upload and Processing State */}
          {!result ? (
            <div className="glass-panel p-8 animate-fade-in" style={{ animationDelay: '0.1s' }}>
              <div 
                className={`upload-zone ${isDragging ? 'dragging' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <input 
                  type="file" 
                  accept=".pdf" 
                  className="hidden-input" 
                  onChange={handleFileSelect}
                  disabled={isExtracting}
                  ref={fileInputRef}
                  title="Upload PDF"
                />
                
                {file ? (
                  <>
                    <File size={64} className="text-brand-primary" />
                    <div>
                      <h3 className="text-xl font-bold mb-1">{file.name}</h3>
                      <p className="text-muted text-sm">{(file.size / 1024 / 1024).toFixed(2)} MB PDF Document</p>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="upload-icon">
                      <UploadCloud size={32} />
                    </div>
                    <div>
                      <h3 className="text-xl font-bold mb-2">Drag & Drop your contract</h3>
                      <p className="text-muted">or click to browse from your computer (PDF only)</p>
                    </div>
                  </>
                )}
              </div>

              {/* Error Message */}
              {error && (
                <div className="mt-6 flex items-start gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400">
                  <AlertCircle size={20} className="mt-0.5 shrink-0" />
                  <div>
                    <strong className="block font-semibold mb-1">Extraction Failed</strong>
                    <span className="text-sm opacity-90">{error}</span>
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="mt-8 flex items-center justify-between">
                <button 
                  className="btn btn-secondary text-sm" 
                  onClick={resetForm}
                  disabled={!file || isExtracting}
                >
                  Clear
                </button>
                <button 
                  className="btn btn-primary" 
                  onClick={handleExtract}
                  disabled={!file || isExtracting}
                >
                  {isExtracting ? (
                    <><Loader2 size={18} className="animate-spin" /> Analyzing Document...</>
                  ) : (
                    <>Run Extraction <ChevronRight size={18} /></>
                  )}
                </button>
              </div>

              {/* Progress Indicator */}
              {isExtracting && (
                <div className="progress-container animate-fade-in">
                  <div className="progress-header">
                    <span className="text-sm font-medium">Running deep learning pipeline...</span>
                    <span className="text-sm font-bold text-brand-primary">{progress}%</span>
                  </div>
                  <div className="progress-bar-bg">
                    <div className="progress-bar-fill" style={{ width: `${progress}%` }}></div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            
            /* Results Presentation */
            <div className="results-container animate-fade-in">
              <div className="flex justify-between items-center mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-green-500/20 flex items-center justify-center text-green-400 border border-green-500/30">
                    <CheckCircle2 size={24} />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold">Extraction Complete</h2>
                    <p className="text-muted text-sm">{result.filename} (ID: {result.document_id.substring(0,8)}...)</p>
                  </div>
                </div>
                <button className="btn btn-secondary" onClick={resetForm}>Extract Another</button>
              </div>

              <div className="results-grid">
                
                {/* Entities Column */}
                <div className="glass-panel p-6">
                  <div className="flex items-center gap-2 mb-6 pb-4 border-b border-white/5">
                    <FileText size={20} className="text-brand-primary" />
                    <h3 className="text-lg font-bold">Extracted Data</h3>
                  </div>

                  {/* Legal Parties */}
                  <div className="entity-group">
                    <div className="entity-header">
                      <Building2 size={16} className="text-blue-400" />
                      <h4 className="text-sm tracking-wider text-muted uppercase font-bold">Legal Parties</h4>
                    </div>
                    <div>
                      {result.entities.PARTY?.length > 0 ? (
                        result.entities.PARTY.map((p, i) => (
                          <span key={i} className="entity-tag tag-party">{p}</span>
                        ))
                      ) : (
                        <span className="text-sm opacity-50 italic px-2">No parties found.</span>
                      )}
                    </div>
                  </div>

                  {/* Dates */}
                  <div className="entity-group">
                    <div className="entity-header">
                      <Calendar size={16} className="text-green-400" />
                      <h4 className="text-sm tracking-wider text-muted uppercase font-bold">Dates (ISO)</h4>
                    </div>
                    <div>
                      {result.entities.DATE?.length > 0 ? (
                        result.entities.DATE.map((d, i) => (
                          <span key={i} className="entity-tag tag-date">{d}</span>
                        ))
                      ) : (
                        <span className="text-sm opacity-50 italic px-2">No dates found.</span>
                      )}
                    </div>
                  </div>

                  {/* Amounts */}
                  <div className="entity-group">
                    <div className="entity-header">
                      <DollarSign size={16} className="text-yellow-400" />
                      <h4 className="text-sm tracking-wider text-muted uppercase font-bold">Monetary Amounts</h4>
                    </div>
                    <div>
                      {result.entities.AMOUNT?.length > 0 ? (
                        result.entities.AMOUNT.map((a, i) => (
                          <span key={i} className="entity-tag tag-amount">{a}</span>
                        ))
                      ) : (
                        <span className="text-sm opacity-50 italic px-2">No amounts found.</span>
                      )}
                    </div>
                  </div>

                  {/* Jurisdictions */}
                  <div className="entity-group mb-0">
                    <div className="entity-header">
                      <MapPin size={16} className="text-pink-400" />
                      <h4 className="text-sm tracking-wider text-muted uppercase font-bold">Jurisdictions</h4>
                    </div>
                    <div>
                      {result.entities.JURISDICTION?.length > 0 ? (
                        result.entities.JURISDICTION.map((j, i) => (
                          <span key={i} className="entity-tag tag-jurisdiction">{j}</span>
                        ))
                      ) : (
                        <span className="text-sm opacity-50 italic px-2">No jurisdictions found.</span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Metrics Column */}
                <div className="flex flex-col gap-6">
                  
                  <div className="glass-panel p-6">
                    <div className="flex items-center gap-2 mb-6 pb-4 border-b border-white/5">
                      <BarChart2 size={20} className="text-brand-secondary" />
                      <h3 className="text-lg font-bold">OCR Metrics</h3>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4">
                      <div className="metric-card">
                        <span className="metric-value">{result.metrics.word_count}</span>
                        <span className="metric-label">Words</span>
                      </div>
                      <div className="metric-card">
                        <span className="metric-value">{result.metrics.text_length}</span>
                        <span className="metric-label">Characters</span>
                      </div>
                      <div className="metric-card">
                        <span className="metric-value">{(result.metrics.noise_ratio * 100).toFixed(1)}%</span>
                        <span className="metric-label">Noise Ratio</span>
                      </div>
                      <div className="metric-card">
                        <span className="metric-value">{(result.metrics.alpha_ratio * 100).toFixed(1)}%</span>
                        <span className="metric-label">Alpha Ratio</span>
                      </div>
                    </div>
                  </div>
                  
                </div>

              </div>
            </div>
          )}

        </div>
      </main>

      {/* Footer */}
      <footer className="mt-auto py-8 border-t border-white/5 text-center text-sm text-muted">
        <p>LexiScan Auto — AI-Powered FinTech Contract Analysis</p>
      </footer>
    </div>
  )
}

export default App
