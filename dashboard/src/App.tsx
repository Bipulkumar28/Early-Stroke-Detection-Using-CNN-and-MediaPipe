import React, { useEffect, useState } from "react";

export default function App() {
  const [liveData, setLiveData] = useState({
    risk: 0,
    mouth: 0,
    eye: 0,
    tilt: 0,
    arm: 0,
    fps: 0,
    alert: 0,
  });

  const [riskHistory, setRiskHistory] = useState<number[]>([]);

  useEffect(() => {
    const fetchData = () => {
      fetch("http://127.0.0.1:5000/live")
        .then((res) => res.json())
        .then((data) => {
          setLiveData(data);

          setRiskHistory((prev) => {
            const updated = [...prev, data.risk];
            return updated.slice(-20);
          });
        })
        .catch((err) => console.log(err));
    };

    fetchData();

    const interval = setInterval(fetchData, 2000);

    return () => clearInterval(interval);
  }, []);

  const features = [
    { name: "Mouth Drop", value: Math.round(liveData.mouth * 100) },
    { name: "Eye Difference", value: Math.round(liveData.eye * 100) },
    { name: "Face Tilt", value: Math.round(liveData.tilt) },
    { name: "Arm Weakness", value: Math.round(liveData.arm * 100) },
  ];

  const maxRisk = Math.max(...riskHistory, 100);

  return (
    <div className="min-h-screen bg-slate-950 text-white p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <div>
          <h1 className="text-4xl font-bold">Stroke Risk Analytics Dashboard</h1>
          <p className="text-slate-400 mt-2">
            Live Stroke Monitoring using AI Detection + Camera Analytics
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          {[
            { title: "Current Risk", value: `${liveData.risk}%` },
            {
              title: "Risk Level",
              value:
                liveData.risk < 30
                  ? "Normal"
                  : liveData.risk < 55
                  ? "Monitor"
                  : liveData.risk < 75
                  ? "Elevated"
                  : "High",
            },
            {
              title: "Alert Status",
              value: liveData.alert ? "ACTIVE" : "SAFE",
            },
            { title: "FPS", value: `${liveData.fps}` },
            { title: "Camera", value: "LIVE" },
          ].map((card) => (
            <div
              key={card.title}
              className="bg-slate-900 rounded-3xl p-5 border border-slate-800 shadow-xl"
            >
              <p className="text-slate-400 text-sm">{card.title}</p>
              <h2 className="text-3xl font-bold mt-3">{card.value}</h2>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-slate-900 rounded-3xl p-6 border border-slate-800">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">Live Risk Trend</h2>
              <span className="text-slate-400 text-sm">Auto Updating</span>
            </div>

            <div className="relative h-72 bg-slate-950 rounded-2xl p-4 overflow-hidden">
              <svg viewBox="0 0 700 250" className="w-full h-full">
                <polyline
                  fill="none"
                  stroke="#38bdf8"
                  strokeWidth="4"
                  points={riskHistory
                    .map((val, index) => {
                      const x = (index / Math.max(riskHistory.length - 1, 1)) * 680;
                      const y = 220 - (val / maxRisk) * 180;
                      return `${x},${y}`;
                    })
                    .join(" ")}
                />
              </svg>
            </div>
          </div>

          <div className="bg-slate-900 rounded-3xl p-6 border border-slate-800 flex flex-col items-center justify-center">
            <h2 className="text-xl font-semibold mb-6">Live Risk Meter</h2>

            <div className="relative w-48 h-48 rounded-full bg-slate-950 flex items-center justify-center border-8 border-orange-500">
              <div className="text-center">
                <div className="text-5xl font-bold">{liveData.risk}%</div>
                <div className="text-orange-400 mt-2">
                  {liveData.alert ? "Alert" : "Monitoring"}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-slate-900 rounded-3xl p-6 border border-slate-800">
            <h2 className="text-xl font-semibold mb-6">Feature Contribution</h2>

            <div className="space-y-5">
              {features.map((feature) => (
                <div key={feature.name}>
                  <div className="flex justify-between mb-2">
                    <span className="text-slate-300">{feature.name}</span>
                    <span>{feature.value}%</span>
                  </div>
                  <div className="w-full h-3 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-cyan-400"
                      style={{ width: `${Math.min(feature.value, 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-slate-900 rounded-3xl p-6 border border-slate-800">
            <h2 className="text-xl font-semibold mb-6">System Status</h2>

            <div className="space-y-4">
              {[
                { label: "Backend Connected", active: true },
                { label: "Camera Detection", active: true },
                { label: "CSV Logging", active: true },
                { label: "Alert Engine", active: liveData.alert === 1 },
              ].map((item) => (
                <div
                  key={item.label}
                  className="flex justify-between items-center bg-slate-950 rounded-xl p-4"
                >
                  <span>{item.label}</span>
                  <div
                    className={`w-3 h-3 rounded-full ${
                      item.active ? "bg-green-400" : "bg-red-400"
                    }`}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
