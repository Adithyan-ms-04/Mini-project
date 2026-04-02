import React, { useState, useRef } from 'react';
import { UploadCloud, CheckCircle2, AlertCircle, RefreshCw, TrendingDown, Activity } from 'lucide-react';
import Docs from './Docs';

export default function App() {
  const [activeTab, setActiveTab] = useState<'predictor' | 'docs'>('predictor');
  const [leftImage, setLeftImage] = useState<File | null>(null);
  const [rightImage, setRightImage] = useState<File | null>(null);
  
  const [clinicalData, setClinicalData] = useState({
    Age: 45, Gender: 0, BMI: 24.5, SystolicBP: 120,
    GFR: 90, SerumCreatinine: 1.0, HbA1c: 5.4, HemoglobinLevels: 14.0,
    BUNLevels: 15.0, ProteinInUrine: 0.1, ACR: 10.0,
    Smoking: 0, AlcoholConsumption: 0, PhysicalActivity: 0, DietQuality: 0, SleepQuality: 0
  });

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const fileInputLeft = useRef<HTMLInputElement>(null);
  const fileInputRight = useRef<HTMLInputElement>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setClinicalData({ ...clinicalData, [name]: parseFloat(value) });
  };

  const handleRunFusion = async () => {
    if (!leftImage) {
      alert("Please upload at least one retinal fundus image.");
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('left_image', leftImage);
      if (rightImage) formData.append('right_image', rightImage);
      formData.append('clinical_data', JSON.stringify(clinicalData));

      const res = await fetch('/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        alert("Error: " + data.error);
        setResults(null);
      } else {
        setResults(data);
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    } catch (err) {
      console.error(err);
      alert("An error occurred during prediction.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResults(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen text-gray-100 font-inter">
      {/* Header */}
      <nav className="glass-card flex items-center justify-between px-8 py-4 mb-8 sticky top-0 z-50 rounded-none border-t-0 border-x-0">
        <div className="text-2xl font-bold tracking-tight bg-gradient-to-r from-blue-400 to-emerald-400 text-transparent bg-clip-text font-outfit cursor-pointer" onClick={() => setActiveTab('predictor')}>
          CKD<span className="text-white">AI</span>
        </div>
        <div className="space-x-6 text-sm font-medium text-gray-300">
          <button 
            onClick={() => setActiveTab('predictor')} 
            className={`transition pb-1 border-b-2 ${activeTab === 'predictor' ? 'text-blue-400 border-blue-400' : 'text-gray-300 border-transparent hover:text-white'}`}
          >
            Predictor
          </button>
          <button 
            onClick={() => setActiveTab('docs')}
            className={`transition pb-1 border-b-2 ${activeTab === 'docs' ? 'text-blue-400 border-blue-400' : 'text-gray-300 border-transparent hover:text-white'}`}
          >
            Docs
          </button>
        </div>
      </nav>

      {activeTab === 'docs' ? (
        <Docs />
      ) : (
        <div className="max-w-7xl mx-auto px-4 pb-16 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <header className="mb-10 text-center">
            <h1 className="text-4xl font-extrabold mb-3 font-outfit">CKD Risk Assessment</h1>
            <p className="text-gray-400 text-lg max-w-2xl mx-auto">
              Multimodal artificial intelligence combining retinal fundus imaging and core clinical biomarkers for early detection.
            </p>
          </header>

          {!results ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-5xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500">
              {/* Clinical Panel */}
              <section className="glass-card p-6 flex flex-col gap-6">
                <div className="flex justify-between items-center border-b border-gray-700/50 pb-4">
                  <h2 className="text-xl font-semibold text-white">Clinical Indicators</h2>
                  <span className="text-xs bg-blue-500/20 text-blue-300 px-3 py-1 rounded-full border border-blue-500/30">Core Features</span>
                </div>

                <form className="flex flex-col gap-5 overflow-y-auto pr-2 custom-scrollbar">
                  <div className="space-y-4">
                    <h3 className="text-sm uppercase tracking-wider text-gray-400 font-semibold mb-2">Key Vitals</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <label className="flex flex-col text-sm">
                        <span className="text-gray-300 mb-1">Age*</span>
                        <input type="number" name="Age" value={clinicalData.Age} onChange={handleInputChange} className="bg-gray-900/50 border border-gray-600 rounded px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 outline-none" required />
                      </label>
                      <label className="flex flex-col text-sm">
                        <span className="text-gray-300 mb-1">Gender*</span>
                        <select name="Gender" value={clinicalData.Gender} onChange={handleInputChange} className="bg-gray-900/50 border border-gray-600 rounded px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 outline-none">
                          <option value={0}>Male</option>
                          <option value={1}>Female</option>
                        </select>
                      </label>
                      <label className="flex flex-col text-sm">
                        <span className="text-gray-300 mb-1">BMI*</span>
                        <input type="number" name="BMI" value={clinicalData.BMI} onChange={handleInputChange} className="bg-gray-900/50 border border-gray-600 rounded px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 outline-none" step="0.1" required />
                      </label>
                      <label className="flex flex-col text-sm">
                        <span className="text-gray-300 mb-1">Systolic BP*</span>
                        <input type="number" name="SystolicBP" value={clinicalData.SystolicBP} onChange={handleInputChange} className="bg-gray-900/50 border border-gray-600 rounded px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 outline-none" required />
                      </label>
                    </div>
                  </div>

                  <div className="space-y-4 border-t border-gray-700/50 pt-5">
                    <h3 className="text-sm uppercase tracking-wider text-gray-400 font-semibold mb-2">Laboratory Metrics</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <label className="flex flex-col text-sm">
                        <span className="text-gray-300 mb-1">GFR*</span>
                        <input type="number" name="GFR" value={clinicalData.GFR} onChange={handleInputChange} className="bg-gray-900/50 border border-gray-600 rounded px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 outline-none" step="0.1" required />
                      </label>
                      <label className="flex flex-col text-sm">
                        <span className="text-gray-300 mb-1">Serum Creatinine*</span>
                        <input type="number" name="SerumCreatinine" value={clinicalData.SerumCreatinine} onChange={handleInputChange} className="bg-gray-900/50 border border-gray-600 rounded px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 outline-none" step="0.1" required />
                      </label>
                      <label className="flex flex-col text-sm">
                        <span className="text-gray-300 mb-1">HbA1c*</span>
                        <input type="number" name="HbA1c" value={clinicalData.HbA1c} onChange={handleInputChange} className="bg-gray-900/50 border border-gray-600 rounded px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 outline-none" step="0.1" required />
                      </label>
                      <label className="flex flex-col text-sm">
                        <span className="text-gray-300 mb-1">Hemoglobin</span>
                        <input type="number" name="HemoglobinLevels" value={clinicalData.HemoglobinLevels} onChange={handleInputChange} className="bg-gray-900/50 border border-gray-600 rounded px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 outline-none" step="0.1" />
                      </label>
                      <label className="flex flex-col text-sm">
                        <span className="text-gray-300 mb-1">BUN Levels</span>
                        <input type="number" name="BUNLevels" value={clinicalData.BUNLevels} onChange={handleInputChange} className="bg-gray-900/50 border border-gray-600 rounded px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 outline-none" step="0.1" />
                      </label>
                      <label className="flex flex-col text-sm">
                        <span className="text-gray-300 mb-1">Protein In Urine</span>
                        <input type="number" name="ProteinInUrine" value={clinicalData.ProteinInUrine} onChange={handleInputChange} className="bg-gray-900/50 border border-gray-600 rounded px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 outline-none" step="0.1" />
                      </label>
                      <label className="flex flex-col text-sm col-span-2">
                        <span className="text-gray-300 mb-1">ACR</span>
                        <input type="number" name="ACR" value={clinicalData.ACR} onChange={handleInputChange} className="bg-gray-900/50 border border-gray-600 rounded px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 outline-none" step="0.1" />
                      </label>
                    </div>
                  </div>

                  <div className="space-y-4 border-t border-gray-700/50 pt-5">
                    <h3 className="text-sm uppercase tracking-wider text-gray-400 font-semibold mb-2">Lifestyle Data</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <label className="flex flex-col text-sm">
                        <span className="text-gray-300 mb-1">Smoking</span>
                        <select name="Smoking" value={clinicalData.Smoking} onChange={handleInputChange} className="bg-gray-900/50 border border-gray-600 rounded px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 outline-none">
                          <option value={0}>Non-smoker</option>
                          <option value={1}>Smoker</option>
                        </select>
                      </label>
                      <label className="flex flex-col text-sm">
                        <span className="text-gray-300 mb-1">Alcohol Consumption</span>
                        <select name="AlcoholConsumption" value={clinicalData.AlcoholConsumption} onChange={handleInputChange} className="bg-gray-900/50 border border-gray-600 rounded px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 outline-none">
                          <option value={0}>None (0)</option>
                          <option value={5}>Low (5 Units)</option>
                          <option value={10}>Moderate (10 Units)</option>
                          <option value={15}>High (15 Units)</option>
                          <option value={20}>Very High (20 Units)</option>
                        </select>
                      </label>
                      <label className="flex flex-col text-sm">
                        <span className="text-gray-300 mb-1">Physical Activity</span>
                        <select name="PhysicalActivity" value={clinicalData.PhysicalActivity} onChange={handleInputChange} className="bg-gray-900/50 border border-gray-600 rounded px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 outline-none">
                          <option value={0}>Sedentary (0/10)</option>
                          <option value={3}>Light Activity (3/10)</option>
                          <option value={5}>Moderate Activity (5/10)</option>
                          <option value={8}>Active (8/10)</option>
                          <option value={10}>Very Active (10/10)</option>
                        </select>
                      </label>
                      <label className="flex flex-col text-sm">
                        <span className="text-gray-300 mb-1">Diet Quality</span>
                        <select name="DietQuality" value={clinicalData.DietQuality} onChange={handleInputChange} className="bg-gray-900/50 border border-gray-600 rounded px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 outline-none">
                          <option value={0}>Poor (0/10)</option>
                          <option value={4}>Fair (4/10)</option>
                          <option value={6}>Average (6/10)</option>
                          <option value={8}>Good (8/10)</option>
                          <option value={10}>Excellent (10/10)</option>
                        </select>
                      </label>
                      <label className="flex flex-col text-sm col-span-2">
                        <span className="text-gray-300 mb-1">Sleep Quality</span>
                        <select name="SleepQuality" value={clinicalData.SleepQuality} onChange={handleInputChange} className="bg-gray-900/50 border border-gray-600 rounded px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 outline-none">
                          <option value={4}>Poor (&lt; 4 Hours/Night)</option>
                          <option value={6}>Fair (~6 Hours/Night)</option>
                          <option value={8}>Good (~8 Hours/Night)</option>
                          <option value={10}>Optimal (10+ Hours/Night)</option>
                        </select>
                      </label>
                    </div>
                  </div>
                </form>
              </section>

              {/* Upload Panel */}
              <section className="glass-card p-6 flex flex-col gap-6">
                <div className="flex justify-between items-center border-b border-gray-700/50 pb-4">
                  <h2 className="text-xl font-semibold text-white">Retinal Imaging</h2>
                  <span className="text-xs bg-emerald-500/20 text-emerald-300 px-3 py-1 rounded-full border border-emerald-500/30">Ocular Branch</span>
                </div>

                <div className="flex-1 flex flex-col gap-4">
                  <div 
                    className="relative border-2 border-dashed border-gray-600 hover:border-blue-400 bg-gray-900/40 rounded-xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition flex-1 group"
                    onClick={() => fileInputLeft.current?.click()}
                  >
                    <input type="file" ref={fileInputLeft} className="hidden" accept="image/*" onChange={(e) => setLeftImage(e.target.files?.[0] || null)} />
                    {leftImage ? (
                      <img src={URL.createObjectURL(leftImage)} alt="Retinal Fundus Image 1" className="absolute inset-0 w-full h-full object-cover rounded-xl" />
                    ) : (
                      <>
                        <UploadCloud className="w-10 h-10 text-gray-400 mb-3 group-hover:text-blue-400 transition" />
                        <p className="text-gray-200 font-medium">Upload Retinal Fundus Image 1</p>
                        <p className="text-xs text-gray-500 mt-1">PNG, JPG up to 10MB</p>
                      </>
                    )}
                  </div>

                  <div 
                    className="relative border-2 border-dashed border-gray-600 hover:border-blue-400 bg-gray-900/40 rounded-xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition flex-1 group"
                    onClick={() => fileInputRight.current?.click()}
                  >
                    <input type="file" ref={fileInputRight} className="hidden" accept="image/*" onChange={(e) => setRightImage(e.target.files?.[0] || null)} />
                    {rightImage ? (
                      <img src={URL.createObjectURL(rightImage)} alt="Retinal Fundus Image 2" className="absolute inset-0 w-full h-full object-cover rounded-xl" />
                    ) : (
                      <>
                        <UploadCloud className="w-10 h-10 text-gray-400 mb-3 group-hover:text-blue-400 transition" />
                        <p className="text-gray-200 font-medium">Upload Retinal Fundus Image 2 (Optional)</p>
                        <p className="text-xs text-gray-500 mt-1">PNG, JPG up to 10MB</p>
                      </>
                    )}
                  </div>
                </div>

                <button 
                  onClick={handleRunFusion} 
                  disabled={loading}
                  className="w-full relative overflow-hidden bg-blue-600 hover:bg-blue-500 text-white font-semibold py-4 rounded-xl shadow-lg transition disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-2 group"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="w-5 h-5 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <div className="absolute inset-0 bg-white/20 block w-0 transform -skew-x-12 group-hover:animate-shine" />
                      Proceed to Analysis
                    </>
                  )}
                </button>
              </section>
            </div>
          ) : (
            /* Results Panel */
            <section className="glass-card p-8 flex flex-col gap-6 relative overflow-hidden max-w-4xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="flex justify-between items-center border-b border-gray-700/50 pb-4">
                <h2 className="text-2xl font-semibold text-white">Prediction Results</h2>
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${results ? 'bg-emerald-500 animate-pulse' : 'bg-gray-500'}`}></span>
                    <span className="text-xs text-gray-400 uppercase tracking-widest">Complete</span>
                  </div>
                  <button onClick={handleReset} className="ml-4 text-xs font-semibold uppercase tracking-wider bg-gray-700 hover:bg-gray-600 text-gray-300 px-4 py-2 rounded-lg transition border border-gray-600">
                    New Prediction
                  </button>
                </div>
              </div>

              <div className="flex flex-col gap-8">
                {/* Score Header */}
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm uppercase tracking-wider mb-1">Final Risk Score</p>
                    <div className="flex items-end gap-2">
                      <span className={`text-5xl font-extrabold pb-0 leading-none ${
                        results.risk_tier.includes('Low') ? 'text-emerald-400' :
                        results.risk_tier.includes('Consultation') ? 'text-yellow-400' : 'text-red-400'
                      }`}>
                        {Math.round(results.final_risk * 100)}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* Sub-Metrics */}
                <div className="space-y-5">
                   <div className="space-y-2">
                     <div className="flex items-center justify-between text-sm">
                       <span className="text-gray-400 group flex items-center gap-2">
                          <div className="w-1.5 h-4 bg-emerald-500 rounded-full" />
                          Ocular Branch Analysis
                       </span>
                       <span className="text-gray-200 font-medium font-mono">{(results.ocular_risk * 100).toFixed(1)}%</span>
                     </div>
                     <div className="h-2 w-full bg-gray-900 rounded-full overflow-hidden">
                       <div className="h-full bg-emerald-500 transition-all duration-1000" style={{ width: `${results.ocular_risk * 100}%`}} />
                     </div>
                   </div>

                   <div className="space-y-2">
                     <div className="flex items-center justify-between text-sm">
                       <span className="text-gray-400 group flex items-center gap-2">
                          <div className="w-1.5 h-4 bg-blue-500 rounded-full" />
                          Clinical Pipeline
                       </span>
                       <span className="text-gray-200 font-medium font-mono">{(results.clinical_risk * 100).toFixed(1)}%</span>
                     </div>
                     <div className="h-2 w-full bg-gray-900 rounded-full overflow-hidden">
                       <div className="h-full bg-blue-500 transition-all duration-1000" style={{ width: `${results.clinical_risk * 100}%`}} />
                     </div>
                   </div>
                </div>

                {/* Tier Card */}
                <div className={`p-5 rounded-xl border border-l-4 shadow-lg ${
                     results.risk_tier.includes('Low') ? 'bg-emerald-500/10 border-emerald-500/30 border-l-emerald-500' :
                     results.risk_tier.includes('Consultation') ? 'bg-yellow-500/10 border-yellow-500/30 border-l-yellow-500' :
                     'bg-red-500/10 border-red-500/30 border-l-red-500'
                }`}>
                   <h3 className="font-semibold text-white mb-2 flex items-center gap-2">
                     {results.risk_tier.includes('Low') ? <CheckCircle2 className="w-5 h-5 text-emerald-400"/> : <AlertCircle className="w-5 h-5 text-current opacity-80"/>}
                     {results.risk_tier}
                   </h3>
                   <p className="text-gray-300 text-sm leading-relaxed">{results.recommendation}</p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-2">
                  {/* Saliency Map */}
                  <div className="flex flex-col gap-4">
                    <div className="flex items-center gap-2">
                      <h3 className="text-lg font-semibold text-white">Attention Map</h3>
                    </div>
                    {results.saliency_map ? (
                      <div className="rounded-xl overflow-hidden border border-gray-700/50 bg-gray-900/50 relative group">
                        <img src={`data:image/png;base64,${results.saliency_map}`} alt="Saliency Map" className="w-full h-auto opacity-90 group-hover:opacity-100 transition" />
                        <div className="absolute top-2 left-2 px-2 py-1 bg-black/60 backdrop-blur-sm rounded text-[10px] uppercase font-semibold text-gray-300 tracking-wider">
                          Grad-CAM++
                        </div>
                      </div>
                    ) : (
                      <div className="flex-1 rounded-xl border border-gray-700/50 bg-gray-900/20 flex flex-col items-center justify-center p-8 text-center min-h-[200px]">
                        <p className="text-gray-500 text-sm">No saliency capability generated</p>
                      </div>
                    )}
                  </div>

                  {/* Lifestyle Simulator */}
                  <div className="flex flex-col gap-4">
                    <div className="flex items-center gap-2">
                      <Activity className="w-5 h-5 text-blue-400" />
                      <h3 className="text-lg font-semibold text-white">Lifestyle Simulator</h3>
                    </div>
                    {results.lifestyle_simulations && results.lifestyle_simulations.length > 0 ? (
                      <div className="bg-gray-900/40 p-5 rounded-xl border border-gray-700/50 h-full flex flex-col">
                        <p className="text-sm text-gray-400 mb-5 leading-relaxed">
                          By optimizing the following specific lifestyle factors, you can dramatically reduce your CKD risk percentage:
                        </p>
                        <div className="space-y-3 flex-1">
                          {results.lifestyle_simulations.map((sim: any, idx: number) => (
                            <div key={idx} className={`p-4 rounded-xl border flex items-center justify-between transition-colors ${sim.is_combined ? 'bg-blue-900/20 border-blue-500/50 mt-4' : 'bg-gray-800/50 border-gray-700 hover:border-blue-500/50'}`}>
                              <div className="flex flex-col flex-1 mr-4">
                                <span className="font-medium text-gray-200">{sim.is_combined ? "Maximum Potential" : sim.factor}</span>
                                <span className="text-xs text-gray-400 mt-1 leading-snug">{sim.action}</span>
                              </div>
                              <div className="flex items-center gap-1.5 text-emerald-400 font-semibold bg-emerald-500/10 px-3 py-1.5 rounded-lg border border-emerald-500/20 whitespace-nowrap">
                                <TrendingDown className="w-4 h-4" />
                                -{sim.reduction_pct.toFixed(1)}%
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="bg-gray-900/40 p-5 rounded-xl border border-gray-700/50 h-full flex flex-col items-center justify-center text-center">
                        <CheckCircle2 className="w-10 h-10 text-emerald-500/50 mb-3" />
                        <p className="text-sm text-gray-300 font-medium tracking-wide">Optimal Lifestyle Maintained</p>
                        <p className="text-xs text-gray-500 mt-2">Your lifestyle inputs are fully optimized. Your core risk arises from other non-modifiable or baseline clinical indicators.</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              {/* Ambient Background decoration */}
              {results && <div className={`absolute -bottom-20 -right-20 w-64 h-64 blur-[100px] rounded-full pointer-events-none opacity-30 ${
                results.risk_tier.includes('Low') ? 'bg-emerald-500' :
                results.risk_tier.includes('Consultation') ? 'bg-yellow-500' : 'bg-red-500'
              }`} />}
            </section>
          )}
        </div>
      )}
    </div>
  );
}
