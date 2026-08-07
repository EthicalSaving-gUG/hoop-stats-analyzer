import React, { useCallback, useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Camera, Play, Pause, SkipBack, SkipForward, FastForward, Maximize } from 'lucide-react';

const VideoPlayer = ({ videoUrl, onCaptureFrame }) => {
    const videoRef = useRef(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [progress, setProgress] = useState(0);
    const [currentTime, setCurrentTime] = useState('00:00');
    const [duration, setDuration] = useState('00:00');
    const [playbackRate, setPlaybackRate] = useState(1);
    const [showSettings, setShowSettings] = useState(false);

    const togglePlay = useCallback(() => {
        const video = videoRef.current;
        if (!video) return;

        setIsPlaying((wasPlaying) => {
            if (wasPlaying) {
                video.pause();
            } else {
                video.play();
            }
            return !wasPlaying;
        });
    }, []);

    const stepFrame = useCallback((direction) => {
        const video = videoRef.current;
        if (!video) return;

        const frameTime = 1 / 30;
        video.currentTime += direction * frameTime;
        video.pause();
        setIsPlaying(false);
    }, []);

    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            
            switch (e.code) {
                case 'Space':
                    e.preventDefault();
                    togglePlay();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    stepFrame(1);
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    stepFrame(-1);
                    break;
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [stepFrame, togglePlay]);

    const formatTime = (timeInSeconds) => {
        if (isNaN(timeInSeconds)) return '00:00';
        const m = Math.floor(timeInSeconds / 60).toString().padStart(2, '0');
        const s = Math.floor(timeInSeconds % 60).toString().padStart(2, '0');
        return `${m}:${s}`;
    };

    const handleTimeUpdate = () => {
        if (!videoRef.current) return;
        const current = videoRef.current.currentTime;
        const dur = videoRef.current.duration;
        setProgress((current / dur) * 100);
        setCurrentTime(formatTime(current));
    };

    const handleLoadedMetadata = () => {
        if (!videoRef.current) return;
        setDuration(formatTime(videoRef.current.duration));
    };

    const skipTime = (seconds) => {
        if (!videoRef.current) return;
        videoRef.current.currentTime += seconds;
    };

    const handleSeek = (e) => {
        if (!videoRef.current) return;
        const seekTime = (e.target.value / 100) * videoRef.current.duration;
        videoRef.current.currentTime = seekTime;
        setProgress(e.target.value);
    };

    const changePlaybackRate = (rate) => {
        if (!videoRef.current) return;
        videoRef.current.playbackRate = rate;
        setPlaybackRate(rate);
        setShowSettings(false);
    };

    const toggleFullscreen = () => {
        if (!videoRef.current) return;
        if (videoRef.current.requestFullscreen) {
            videoRef.current.requestFullscreen();
        }
    };

    const captureFrame = () => {
        if (!videoRef.current || !onCaptureFrame) return;
        onCaptureFrame({
            videoUrl,
            timestamp: videoRef.current.currentTime || 0,
        });
    };

    return (
        <div className="relative group w-full bg-slate-900 rounded-xl overflow-hidden border border-slate-700/50 flex flex-col h-full">
            <div className="relative flex-grow bg-black flex items-center justify-center">
                {!videoUrl ? (
                    <div className="text-slate-500 flex flex-col items-center">
                        <Play className="w-12 h-12 mb-2 opacity-50" />
                        <p>Kein Video geladen</p>
                    </div>
                ) : (
                    <video
                        ref={videoRef}
                        src={videoUrl}
                        className="w-full h-full object-contain"
                        onTimeUpdate={handleTimeUpdate}
                        onLoadedMetadata={handleLoadedMetadata}
                        onClick={togglePlay}
                    />
                )}
            </div>

            {/* Controls Overlay */}
            <div className="bg-slate-900/90 p-4 border-t border-slate-700/50 shrink-0">
                {/* Seekbar */}
                <div className="flex items-center gap-3 mb-3">
                    <span className="text-xs text-slate-400 font-mono w-10 text-right">{currentTime}</span>
                    <input
                        type="range"
                        min="0"
                        max="100"
                        value={progress || 0}
                        onChange={handleSeek}
                        className="flex-grow h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-orange-500"
                        disabled={!videoUrl}
                    />
                    <span className="text-xs text-slate-400 font-mono w-10">{duration}</span>
                </div>

                {/* Buttons */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <button onClick={() => skipTime(-5)} disabled={!videoUrl} className="text-slate-300 hover:text-white transition disabled:opacity-50">
                            <SkipBack className="w-5 h-5" />
                        </button>
                        <button onClick={() => stepFrame(-1)} disabled={!videoUrl} className="text-slate-400 hover:text-white transition disabled:opacity-50 text-xs font-bold px-2 py-1 bg-slate-800 rounded">
                            -1F
                        </button>
                        
                        <button onClick={togglePlay} disabled={!videoUrl} className="w-10 h-10 flex items-center justify-center bg-orange-500 hover:bg-orange-600 text-white rounded-full transition disabled:opacity-50">
                            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-1" />}
                        </button>

                        <button onClick={() => stepFrame(1)} disabled={!videoUrl} className="text-slate-400 hover:text-white transition disabled:opacity-50 text-xs font-bold px-2 py-1 bg-slate-800 rounded">
                            +1F
                        </button>
                        <button onClick={() => skipTime(5)} disabled={!videoUrl} className="text-slate-300 hover:text-white transition disabled:opacity-50">
                            <SkipForward className="w-5 h-5" />
                        </button>
                    </div>

                    <div className="flex items-center gap-4 relative">
                        {onCaptureFrame && (
                            <button onClick={captureFrame} disabled={!videoUrl} className="text-slate-300 hover:text-white transition disabled:opacity-50" title="Frame fuer Grounding extrahieren">
                                <Camera className="w-5 h-5" />
                            </button>
                        )}
                        <div className="relative">
                            <button 
                                onClick={() => setShowSettings(!showSettings)} 
                                disabled={!videoUrl}
                                className="flex items-center gap-1 text-slate-300 hover:text-white transition disabled:opacity-50 text-sm font-semibold bg-slate-800 px-2 py-1 rounded"
                            >
                                <FastForward className="w-4 h-4" /> {playbackRate}x
                            </button>
                            {showSettings && (
                                <div className="absolute bottom-full right-0 mb-2 bg-slate-800 border border-slate-700 rounded-lg shadow-xl overflow-hidden z-50">
                                    {[0.25, 0.5, 1, 1.5, 2].map(rate => (
                                        <button
                                            key={rate}
                                            onClick={() => changePlaybackRate(rate)}
                                            className={`block w-full text-left px-4 py-2 text-sm hover:bg-slate-700 transition ${playbackRate === rate ? 'text-orange-400 font-bold' : 'text-slate-300'}`}
                                        >
                                            {rate}x
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                        <button onClick={toggleFullscreen} disabled={!videoUrl} className="text-slate-300 hover:text-white transition disabled:opacity-50">
                            <Maximize className="w-5 h-5" />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default VideoPlayer;
