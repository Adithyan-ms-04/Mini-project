import { BookOpen, Users, BrainCircuit, ShieldCheck, Database, Server } from 'lucide-react';

export default function Docs() {
  const teamMembers = [
    {
      name: "Adithyan M S",
      role: "Lead AI Architect",
      contribution: "Designed and implemented the dual-branch Late-Fusion architecture, combining Random Forest and EfficientNet-B3. Handled the Weighted Reliability Score (WRS) ensemble logic.",
      icon: <BrainCircuit className="w-8 h-8 text-blue-400" />
    },
    {
      name: "Alan Seby",
      role: "Frontend Developer & UI/UX Designer",
      contribution: "Built the React + Tailwind CSS dashboard. Engineered the interactive glassmorphism UI, real-time risk gauge, and responsive data visualizations.",
      icon: <BookOpen className="w-8 h-8 text-emerald-400" />
    },
    {
      name: "Haripriya B",
      role: "Backend Optimization & Data Pipeline",
      contribution: "Developed the FastAPI backend infrastructure. Optimized image transformation sequences (Graham transforms) and managed the API-to-Model continuous routing.",
      icon: <Server className="w-8 h-8 text-purple-400" />
    },
    {
      name: "Lakshmipriya S",
      role: "Clinical Data & Guardrails Engineer",
      contribution: "Led the clinical feature pruning (top 10 biomarkers). Engineered the InputGuardrail system containing OOD validation, anomaly detection, and probability calibration.",
      icon: <ShieldCheck className="w-8 h-8 text-yellow-400" />
    }
  ];

  return (
    <div className="max-w-5xl mx-auto px-4 pb-16 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <header className="mb-12 text-center">
        <h1 className="text-4xl font-extrabold mb-4 font-outfit">Project Documentation</h1>
        <p className="text-gray-400 text-lg max-w-2xl mx-auto">
          Technical specifications, architecture details, and comprehensive credits for the CKD AI Diagnostic System.
        </p>
      </header>

      {/* Team Credits Section */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-8 border-b border-gray-700/50 pb-4">
          <Users className="w-6 h-6 text-blue-400" />
          <h2 className="text-2xl font-semibold text-white">Development Team & Credits</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {teamMembers.map((member, idx) => (
            <div key={idx} className="glass-card p-6 border-l-4 border-l-blue-500/50 hover:border-l-blue-400 transition cursor-default group">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-gray-900/50 rounded-xl group-hover:scale-110 transition-transform">
                  {member.icon}
                </div>
                <div>
                  <h3 className="text-xl font-bold text-gray-100 mb-1">{member.name}</h3>
                  <p className="text-sm text-blue-400 font-medium mb-3">{member.role}</p>
                  <p className="text-gray-400 text-sm leading-relaxed">
                    {member.contribution}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Architecture Overview */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-8 border-b border-gray-700/50 pb-4">
          <Database className="w-6 h-6 text-emerald-400" />
          <h2 className="text-2xl font-semibold text-white">System Architecture Overview</h2>
        </div>
        
        <div className="glass-card p-8 space-y-6 text-gray-300 leading-relaxed">
          <p>
            The <strong>CKD AI System</strong> is a highly advanced multimodal framework designed to assess the risk of Chronic Kidney Disease (CKD) by fusing two distinct diagnostic branches:
          </p>
          <ul className="list-disc pl-6 space-y-3">
            <li>
              <strong className="text-white">The Clinical Branch:</strong> A highly regularized, probability-calibrated Random Forest model. It processes 10 critical biological metrics (including GFR, Serum Creatinine, and ACR) to determine baseline organ function. Features an Input Guardrail to reject physically impossible or Out-Of-Distribution (OOD) telemetry.
            </li>
            <li>
              <strong className="text-white">The Ocular Branch:</strong> An EfficientNet-B3 Convolutional Neural Network trained on Retinal Fundus imagery. It utilizes a computationally expensive Graham Transform to isolate microvascular damage indicative of systemic hypertensive or diabetic nephropathy.
            </li>
            <li>
              <strong className="text-white">Late-Fusion Ensemble logic:</strong> A Weighted Reliability Score (WRS) dynamically balances the sensitivity of the ocular branch against the specificity of the clinical branch, stabilizing edge-case anomalies to output a Three-Tier Risk classification metric.
            </li>
          </ul>
        </div>
      </section>
    </div>
  );
}
