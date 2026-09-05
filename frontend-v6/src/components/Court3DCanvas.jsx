import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

const Court3DCanvas = ({ courtTracking = null, shots = [] }) => {
    const mountRef = useRef(null);
    const sceneRef = useRef(null);
    const rendererRef = useRef(null);
    const cameraRef = useRef(null);
    const playersGroupRef = useRef(null);
    const ballMeshRef = useRef(null);
    const shotArcsGroupRef = useRef(null);
    const heatmapGroupRef = useRef(null);
    const isDraggingRef = useRef(false);
    const previousMousePositionRef = useRef({ x: 0, y: 0 });
    const cameraAngleRef = useRef({ theta: Math.PI / 4, phi: Math.PI / 6, radius: 35 });
    
    const [cameraPreset, setCameraPreset] = useState('broadcast');
    const [showHeatmap, setShowHeatmap] = useState(false);

    const updateCameraPosition = () => {
        if (!cameraRef.current) return;
        const { theta, phi, radius } = cameraAngleRef.current;
        const x = radius * Math.sin(phi) * Math.sin(theta);
        const y = radius * Math.cos(phi);
        const z = radius * Math.sin(phi) * Math.cos(theta);
        cameraRef.current.position.set(x, Math.max(y, 2), z);
        cameraRef.current.lookAt(0, 0, 0);
    };

    const setPreset = (preset) => {
        setCameraPreset(preset);
        if (preset === 'broadcast') {
            cameraAngleRef.current = { theta: 0, phi: Math.PI / 3.2, radius: 28 };
        } else if (preset === 'overhead') {
            cameraAngleRef.current = { theta: 0, phi: 0.05, radius: 32 };
        } else if (preset === 'baseline') {
            cameraAngleRef.current = { theta: Math.PI / 2, phi: Math.PI / 3.5, radius: 26 };
        }
        updateCameraPosition();
    };

    useEffect(() => {
        const container = mountRef.current;
        if (!container) return;

        const width = container.clientWidth || 800;
        const height = container.clientHeight || 450;

        // Scene
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0f1d);
        scene.fog = new THREE.FogExp2(0x0a0f1d, 0.015);
        sceneRef.current = scene;

        // Camera
        const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        cameraRef.current = camera;
        updateCameraPosition();

        // Renderer
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        rendererRef.current = renderer;
        container.appendChild(renderer.domElement);

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
        scene.add(ambientLight);

        const stadiumLight1 = new THREE.DirectionalLight(0xffeedd, 1.2);
        stadiumLight1.position.set(15, 25, 15);
        stadiumLight1.castShadow = true;
        stadiumLight1.shadow.mapSize.width = 1024;
        stadiumLight1.shadow.mapSize.height = 1024;
        scene.add(stadiumLight1);

        const stadiumLight2 = new THREE.DirectionalLight(0x88bbff, 0.8);
        stadiumLight2.position.set(-15, 25, -15);
        scene.add(stadiumLight2);

        // Court Floor (28m x 15m FIBA / NBA scale)
        const courtGeo = new THREE.PlaneGeometry(28, 15);
        const courtMat = new THREE.MeshStandardMaterial({
            color: 0xd97736,
            roughness: 0.35,
            metalness: 0.1,
        });
        const courtFloor = new THREE.Mesh(courtGeo, courtMat);
        courtFloor.rotation.x = -Math.PI / 2;
        courtFloor.receiveShadow = true;
        scene.add(courtFloor);

        // Court Border / Apron
        const apronGeo = new THREE.PlaneGeometry(32, 19);
        const apronMat = new THREE.MeshStandardMaterial({
            color: 0x1e293b,
            roughness: 0.6,
        });
        const apron = new THREE.Mesh(apronGeo, apronMat);
        apron.rotation.x = -Math.PI / 2;
        apron.position.y = -0.01;
        scene.add(apron);

        // Court Line helper
        const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 });

        const drawCourtLine = (points) => {
            const geo = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geo, lineMat);
            line.position.y = 0.01;
            scene.add(line);
        };

        // Perimeter
        drawCourtLine([
            new THREE.Vector3(-14, 0, -7.5),
            new THREE.Vector3(14, 0, -7.5),
            new THREE.Vector3(14, 0, 7.5),
            new THREE.Vector3(-14, 0, 7.5),
            new THREE.Vector3(-14, 0, -7.5),
        ]);

        // Half-court line
        drawCourtLine([
            new THREE.Vector3(0, 0, -7.5),
            new THREE.Vector3(0, 0, 7.5),
        ]);

        // Center circle (radius 1.8m)
        const centerCirclePoints = [];
        for (let i = 0; i <= 64; i++) {
            const theta = (i / 64) * Math.PI * 2;
            centerCirclePoints.push(new THREE.Vector3(Math.cos(theta) * 1.8, 0, Math.sin(theta) * 1.8));
        }
        drawCourtLine(centerCirclePoints);

        // 3D Hoops & Backboards (Left & Right)
        const createHoop = (side) => {
            const hoopGroup = new THREE.Group();
            const poleX = side * 14.8;
            const rimX = side * 12.425;

            // Pole
            const poleGeo = new THREE.CylinderGeometry(0.12, 0.12, 4, 16);
            const poleMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.8 });
            const pole = new THREE.Mesh(poleGeo, poleMat);
            pole.position.set(poleX, 2, 0);
            hoopGroup.add(pole);

            // Boom arm
            const armGeo = new THREE.CylinderGeometry(0.08, 0.08, 2.5, 16);
            const arm = new THREE.Mesh(armGeo, poleMat);
            arm.rotation.z = side * Math.PI / 4;
            arm.position.set(poleX - side * 0.9, 3.5, 0);
            hoopGroup.add(arm);

            // Backboard
            const bbGeo = new THREE.BoxGeometry(0.05, 1.05, 1.8);
            const bbMat = new THREE.MeshStandardMaterial({
                color: 0xffffff,
                transparent: true,
                opacity: 0.85,
                roughness: 0.1
            });
            const backboard = new THREE.Mesh(bbGeo, bbMat);
            backboard.position.set(side * 12.8, 3.3, 0);
            hoopGroup.add(backboard);

            // Rim (Orange Ring at height 3.05m)
            const rimGeo = new THREE.TorusGeometry(0.225, 0.02, 16, 32);
            const rimMat = new THREE.MeshStandardMaterial({ color: 0xf97316, metalness: 0.5, roughness: 0.3 });
            const rim = new THREE.Mesh(rimGeo, rimMat);
            rim.rotation.x = Math.PI / 2;
            rim.position.set(rimX, 3.05, 0);
            hoopGroup.add(rim);

            return hoopGroup;
        };

        scene.add(createHoop(1));
        scene.add(createHoop(-1));

        // Group for dynamic players
        const playersGroup = new THREE.Group();
        scene.add(playersGroup);
        playersGroupRef.current = playersGroup;

        // Dynamic 3D Ball
        const ballGeo = new THREE.SphereGeometry(0.12, 24, 24);
        const ballMat = new THREE.MeshStandardMaterial({
            color: 0xf97316,
            roughness: 0.4,
            metalness: 0.1,
            emissive: 0xea580c,
            emissiveIntensity: 0.2
        });
        const ballMesh = new THREE.Mesh(ballGeo, ballMat);
        ballMesh.castShadow = true;
        ballMesh.position.set(0, 0.12, 0);
        scene.add(ballMesh);
        ballMeshRef.current = ballMesh;

        // Group for dynamic Shot Arcs
        const shotArcsGroup = new THREE.Group();
        scene.add(shotArcsGroup);
        shotArcsGroupRef.current = shotArcsGroup;

        // Group for dynamic 3D Heatmap
        const heatmapGroup = new THREE.Group();
        scene.add(heatmapGroup);
        heatmapGroupRef.current = heatmapGroup;

        // Animation Loop
        let animationFrameId;
        const animate = () => {
            animationFrameId = requestAnimationFrame(animate);
            renderer.render(scene, camera);
        };
        animate();

        // Mouse Controls (Orbit / Pan)
        const handleMouseDown = (e) => {
            isDraggingRef.current = true;
            previousMousePositionRef.current = { x: e.clientX, y: e.clientY };
        };

        const handleMouseMove = (e) => {
            if (!isDraggingRef.current) return;
            const deltaX = e.clientX - previousMousePositionRef.current.x;
            const deltaY = e.clientY - previousMousePositionRef.current.y;

            cameraAngleRef.current.theta -= deltaX * 0.008;
            cameraAngleRef.current.phi = Math.max(0.1, Math.min(Math.PI / 2 - 0.05, cameraAngleRef.current.phi - deltaY * 0.008));

            updateCameraPosition();
            previousMousePositionRef.current = { x: e.clientX, y: e.clientY };
        };

        const handleMouseUp = () => {
            isDraggingRef.current = false;
        };

        const handleWheel = (e) => {
            cameraAngleRef.current.radius = Math.max(10, Math.min(60, cameraAngleRef.current.radius + e.deltaY * 0.03));
            updateCameraPosition();
        };

        const domElem = renderer.domElement;
        domElem.addEventListener('mousedown', handleMouseDown);
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
        domElem.addEventListener('wheel', handleWheel, { passive: true });

        // Resize handler
        const handleResize = () => {
            if (!container || !renderer || !camera) return;
            const newW = container.clientWidth;
            const newH = container.clientHeight;
            camera.aspect = newW / newH;
            camera.updateProjectionMatrix();
            renderer.setSize(newW, newH);
        };
        window.addEventListener('resize', handleResize);

        return () => {
            cancelAnimationFrame(animationFrameId);
            window.removeEventListener('resize', handleResize);
            domElem.removeEventListener('mousedown', handleMouseDown);
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
            domElem.removeEventListener('wheel', handleWheel);
            if (renderer.domElement && container.contains(renderer.domElement)) {
                container.removeChild(renderer.domElement);
            }
            renderer.dispose();
        };
    }, []);

    // Update dynamic player cylinders & ball from real-time courtTracking
    useEffect(() => {
        if (!playersGroupRef.current || !ballMeshRef.current) return;

        // Clear existing player meshes
        while (playersGroupRef.current.children.length > 0) {
            const obj = playersGroupRef.current.children[0];
            playersGroupRef.current.remove(obj);
        }

        if (courtTracking && courtTracking.players) {
            Object.entries(courtTracking.players).forEach(([playerId, pData]) => {
                const pos = pData.pos; // [mx, my] in court coordinates (0 to 28m, 0 to 15m)
                if (!pos || pos.length < 2) return;

                // Center court coordinates around (0, 0)
                const cx = pos[0] - 14.0;
                const cz = pos[1] - 7.5;
                const team = pData.team || 'home';

                const pGroup = new THREE.Group();

                // Player cylinder (height 1.95m, radius 0.35m)
                const cylGeo = new THREE.CylinderGeometry(0.35, 0.35, 1.8, 16);
                const isHome = team === 'home';
                const cylMat = new THREE.MeshStandardMaterial({
                    color: isHome ? 0x3b82f6 : 0x10b981,
                    metalness: 0.2,
                    roughness: 0.5,
                });
                const cyl = new THREE.Mesh(cylGeo, cylMat);
                cyl.position.y = 0.9;
                cyl.castShadow = true;
                pGroup.add(cyl);

                // Head sphere
                const headGeo = new THREE.SphereGeometry(0.22, 16, 16);
                const headMat = new THREE.MeshStandardMaterial({ color: 0xfde047 });
                const head = new THREE.Mesh(headGeo, headMat);
                head.position.y = 1.95;
                pGroup.add(head);

                pGroup.position.set(cx, 0, cz);
                playersGroupRef.current.add(pGroup);
            });
        }

        // Update ball position
        if (courtTracking && courtTracking.ball && courtTracking.ball.length >= 2) {
            const bx = courtTracking.ball[0] - 14.0;
            const bz = courtTracking.ball[1] - 7.5;
            const by = courtTracking.ball.length > 2 ? courtTracking.ball[2] : 0.8;
            ballMeshRef.current.position.set(bx, Math.max(0.12, by), bz);
        }
    }, [courtTracking]);

    // Update Shot Trajectory Arcs
    useEffect(() => {
        if (!shotArcsGroupRef.current) return;

        while (shotArcsGroupRef.current.children.length > 0) {
            const obj = shotArcsGroupRef.current.children[0];
            shotArcsGroupRef.current.remove(obj);
        }

        shots.forEach((shot) => {
            if (!shot.coord_m || shot.coord_m.length < 2) return;
            const sx = shot.coord_m[0] - 14.0;
            const sz = shot.coord_m[1] - 7.5;
            const targetX = sx > 0 ? 12.425 : -12.425;
            const targetZ = 0;
            const targetY = 3.05;

            // Compute quadratic bezier curve for parabolic shot
            const midX = (sx + targetX) / 2;
            const midZ = (sz + targetZ) / 2;
            const midY = Math.max(targetY + 2.5, 5.5);

            const curve = new THREE.QuadraticBezierCurve3(
                new THREE.Vector3(sx, 1.8, sz),
                new THREE.Vector3(midX, midY, midZ),
                new THREE.Vector3(targetX, targetY, targetZ)
            );

            const pts = curve.getPoints(32);
            const geo = new THREE.BufferGeometry().setFromPoints(pts);
            const mat = new THREE.LineBasicMaterial({
                color: shot.status === 'made' ? 0x10b981 : 0xf43f5e,
                linewidth: 2,
                transparent: true,
                opacity: 0.85
            });
            const line = new THREE.Line(geo, mat);
            shotArcsGroupRef.current.add(line);
        });
    }, [shots]);

    // Update 3D Shot Heatmap Layer on the court floor
    useEffect(() => {
        if (!heatmapGroupRef.current) return;

        while (heatmapGroupRef.current.children.length > 0) {
            const obj = heatmapGroupRef.current.children[0];
            heatmapGroupRef.current.remove(obj);
        }

        if (!showHeatmap) return;

        shots.forEach((shot) => {
            if (!shot.coord_m || shot.coord_m.length < 2) return;
            const sx = shot.coord_m[0] - 14.0;
            const sz = shot.coord_m[1] - 7.5;
            const isMade = shot.status === 'made';

            // Radial gradient spot on court floor
            const circleGeo = new THREE.CircleGeometry(1.2, 24);
            const circleMat = new THREE.MeshBasicMaterial({
                color: isMade ? 0x10b981 : 0xf43f5e,
                transparent: true,
                opacity: 0.35,
                depthWrite: false,
                blending: THREE.AdditiveBlending,
            });
            const circleMesh = new THREE.Mesh(circleGeo, circleMat);
            circleMesh.rotation.x = -Math.PI / 2;
            circleMesh.position.set(sx, 0.02, sz);
            heatmapGroupRef.current.add(circleMesh);
        });
    }, [shots, showHeatmap]);

    return (
        <div className="relative w-full h-[420px] rounded-2xl overflow-hidden bg-slate-950 border border-slate-800/80 shadow-2xl">
            <div ref={mountRef} className="w-full h-full cursor-grab active:cursor-grabbing" />
            
            {/* Overlay Camera & Heatmap Controls */}
            <div className="absolute top-4 left-4 flex flex-wrap items-center gap-2 bg-slate-900/85 backdrop-blur-md px-3 py-1.5 rounded-xl border border-slate-700/60 shadow-lg text-xs font-medium">
                <span className="text-slate-400">3D Kamera:</span>
                <button
                    onClick={() => setPreset('broadcast')}
                    className={`px-2.5 py-1 rounded-lg transition-colors ${cameraPreset === 'broadcast' ? 'bg-indigo-600 text-white font-semibold' : 'text-slate-300 hover:bg-slate-800'}`}
                >
                    Broadcast
                </button>
                <button
                    onClick={() => setPreset('overhead')}
                    className={`px-2.5 py-1 rounded-lg transition-colors ${cameraPreset === 'overhead' ? 'bg-indigo-600 text-white font-semibold' : 'text-slate-300 hover:bg-slate-800'}`}
                >
                    Overhead
                </button>
                <button
                    onClick={() => setPreset('baseline')}
                    className={`px-2.5 py-1 rounded-lg transition-colors ${cameraPreset === 'baseline' ? 'bg-indigo-600 text-white font-semibold' : 'text-slate-300 hover:bg-slate-800'}`}
                >
                    Baseline
                </button>
                <div className="w-[1px] h-4 bg-slate-700 mx-1" />
                <button
                    onClick={() => setShowHeatmap(!showHeatmap)}
                    className={`px-2.5 py-1 rounded-lg flex items-center gap-1.5 transition-all ${showHeatmap ? 'bg-orange-500 text-white font-bold shadow-lg shadow-orange-500/25' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'}`}
                >
                    <span>🔥 3D Heatmap</span>
                    <span className="text-[10px] opacity-75">{showHeatmap ? 'AN' : 'AUS'}</span>
                </button>
            </div>

            {/* Orbit Instructions Badge */}
            <div className="absolute bottom-3 right-4 text-[10px] font-mono text-slate-500 bg-slate-950/60 backdrop-blur-sm px-2 py-1 rounded border border-slate-800/50 pointer-events-none">
                Maus gedrückt halten zum Schwenken • Scrollen zum Zoomen
            </div>
        </div>
    );
};

export default Court3DCanvas;
