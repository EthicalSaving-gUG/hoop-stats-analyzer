import React, { useState } from 'react';
import { Upload, Link, BarChart3, Activity, Cpu, Play, CheckCircle2, AlertCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

const App = () => {
    const [file, setFile] = useState(null);
    const [streamUrl, setStreamUrl] = useState('');
    const [status, setStatus] = useState('idle'); // idle, processing, complete
    const [activeTab, setActiveTab] = useState('upload'); // upload, stream

    const handleUpload = async () => {
        if (!file) return;
        setStatus('processing');
        const formData = new FormData();
        formData.append('file', file);
        try {
            await axios.post('http://localhost:8000/analyze/file', formData);
            setStatus('complete');
        } catch (error) {
            console.error(error);
            setStatus('idle');
        }
    };

    const handleStream = async () => {
        if (!streamUrl) return;
        setStatus('processing');
        try {
            await axios.post('http://localhost:8000/analyze/stream', { url: streamUrl });
            setStatus('processing'); // Stays in processing for stream
        } catch (error) {
            console.error(error);
            setStatus('idle');
        }
    };

    return (
        <div className="min-h-screen gradient-bg p-8 flex flex-col items-center justify-center relative">
            {/* Background Orbs */}
            <div className="absolute top-0 left-0 w-96 h-96 bg-blue-600/10 rounded-full blur-[128px] animate-pulse-slow"></div>
            <div className="absolute bottom-0 right-0 w-96 h-96 bg-purple-600/10 rounded-full blur-[128px] animate-pulse-slow"></div>

            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="w-full max-w-4xl z-10"
            >
                <header className="mb-12 text-center">
                    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-sm font-semibold mb-4">
                        <Activity size={16} />
                        Basketball Computer Vision
                    </div>
                    <h1 className="text-6xl font-bold mb-4 tracking-tight">Hoop Stats <span className="text-blue-500">Analyzer</span></h1>
                    <p className="text-slate-400 text-lg max-w-xl mx-auto">
                        Extract professional-grade player metrics from game footage using YOLOv11 and Homography mapping.
                    </p>
                </header>

                <div className="glass p-1">
                    <nav className="flex gap-2 p-2">
                        <button
                            onClick={() => setActiveTab('upload')}
                            className={`flex-1 flex items-center justify-center gap-2 py-4 rounded-xl font-semibold transition-all ${activeTab === 'upload' ? 'bg-white/10 text-white shadow-xl' : 'text-slate-500 hover:text-slate-300 hover:bg-white/5'}`}
                        >
                            <Upload size={20} />
                            Video Upload
                        </button>
                        <button
                            onClick={() => setActiveTab('stream')}
                            className={`flex-1 flex items-center justify-center gap-2 py-4 rounded-xl font-semibold transition-all ${activeTab === 'stream' ? 'bg-white/10 text-white shadow-xl' : 'text-slate-500 hover:text-slate-300 hover:bg-white/5'}`}
                        >
                            <Link size={20} />
                            Stream Link
                        </button>
                    </nav>

                    <div className="p-8">
                        <AnimatePresence mode="wait">
                            {activeTab === 'upload' ? (
                                <motion.div
                                    key="upload"
                                    initial={{ opacity: 0, x: -10 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: 10 }}
                                    className="space-y-6"
                                >
                                    <label className="block w-full cursor-pointer group">
                                        <div className="border-2 border-dashed border-white/10 rounded-3xl p-12 flex flex-col items-center justify-center group-hover:border-blue-500/50 transition-all bg-white/5">
                                            <div className="w-16 h-16 bg-blue-500/20 rounded-2xl flex items-center justify-center text-blue-500 mb-6 group-hover:scale-110 transition-transform">
                                                <Upload size={32} />
                                            </div>
                                            <span className="text-xl font-semibold mb-2">{file ? file.name : "Select Game Video"}</span>
                                            <span className="text-slate-500">MP4, MOV up to 500MB</span>
                                            <input type="file" className="hidden" onChange={(e) => setFile(e.target.files[0])} />
                                        </div>
                                    </label>

                                    <button
                                        disabled={!file || status === 'processing'}
                                        onClick={handleUpload}
                                        className="w-full btn-primary py-5 rounded-3xl text-xl font-bold flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        {status === 'processing' ? <Cpu className="animate-spin" /> : <Play />}
                                        {status === 'processing' ? 'Analyzing Game...' : 'Begin Analysis'}
                                    </button>
                                </motion.div>
                            ) : (
                                <motion.div
                                    key="stream"
                                    initial={{ opacity: 0, x: -10 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: 10 }}
                                    className="space-y-6"
                                >
                                    <div className="bg-white/5 border border-white/10 rounded-3xl p-1 py-1 flex items-center">
                                        <div className="px-6 text-slate-500">
                                            <Link size={24} />
                                        </div>
                                        <input
                                            type="text"
                                            placeholder="Enter RTSP or HTTP Stream URL"
                                            className="bg-transparent border-none outline-none flex-1 py-6 text-xl"
                                            value={streamUrl}
                                            onChange={(e) => setStreamUrl(e.target.value)}
                                        />
                                    </div>

                                    <button
                                        disabled={!streamUrl || status === 'processing'}
                                        onClick={handleStream}
                                        className="w-full btn-primary py-5 rounded-3xl text-xl font-bold flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        {status === 'processing' ? <Cpu className="animate-spin" /> : <Play />}
                                        {status === 'processing' ? 'Connecting to Stream...' : 'Start Live Tracking'}
                                    </button>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </div>

                {/* Status Section */}
                <AnimatePresence>
                    {status !== 'idle' && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="mt-8 glass p-8 grid grid-cols-1 md:grid-cols-3 gap-8"
                        >
                            <div className="flex items-center gap-4">
                                <div className={`p-4 rounded-2xl ${status === 'complete' ? 'bg-green-500/20 text-green-500' : 'bg-blue-500/20 text-blue-500'}`}>
                                    {status === 'complete' ? <CheckCircle2 size={24} /> : <BarChart3 size={24} />}
                                </div>
                                <div>
                                    <h3 className="font-semibold text-slate-300">Phase</h3>
                                    <p className="text-xl font-bold">{status === 'complete' ? 'Export Ready' : 'Processing AI'}</p>
                                </div>
                            </div>

                            <div className="flex items-center gap-4">
                                <div className="p-4 rounded-2xl bg-purple-500/20 text-purple-500">
                                    <Cpu size={24} />
                                </div>
                                <div>
                                    <h3 className="font-semibold text-slate-300">Model</h3>
                                    <p className="text-xl font-bold">YOLOv11 Pro</p>
                                </div>
                            </div>

                            <div className="flex items-center gap-4">
                                <div className="p-4 rounded-2xl bg-orange-500/20 text-orange-500">
                                    <Activity size={24} />
                                </div>
                                <div>
                                    <h3 className="font-semibold text-slate-300">Tracking</h3>
                                    <p className="text-xl font-bold">ByteTrack V2</p>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                <footer className="mt-12 flex justify-between items-center text-slate-500 px-4">
                    <p>© 2026 Hoop Stats CV Engine</p>
                    <div className="flex gap-6">
                        <span className="flex items-center gap-2"><div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div> Backend Online</span>
                    </div>
                </footer>
            </motion.div>
        </div>
    );
};

export default App;
