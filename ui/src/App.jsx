import React, {useEffect, useMemo, useState} from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import SettingsPage from './pages/SettingsPage';
import {API_URL, fetchJson} from './utils/api';
import {Loader} from 'lucide-react';

export default function App() {
    const [devices, setDevices] = useState([]);
    const [allActuators, setAllActuators] = useState([]);
    const [activePage, setActivePage] = useState('dashboard');
    const [activeDeviceId, setActiveDeviceId] = useState(null);
    const activeDeviceIdRef = React.useRef(activeDeviceId);
    const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString('it-IT'));
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [bannerError, setBannerError] = useState(null);

    // Hysteresis globale per attuatore (UI-only per ora)
    const [hysteresis, setHysteresis] = useState({}); // { [actuatorName]: {on, off} }

    const [hasLoadedOnce, setHasLoadedOnce] = useState(false);
    const failCountRef = React.useRef(0);
    const fetchCtrlRef = React.useRef(null);

    useEffect(() => {
        activeDeviceIdRef.current = activeDeviceId;
    }, [activeDeviceId]);

    const pickDefaultActive = (list, previousId) => {
        const exists = list.some(d => d.device_id === previousId);
        if (exists) return previousId;
        const firstAssigned = list.find(d => d.room_name);
        return firstAssigned ? firstAssigned.device_id : (list[0]?.device_id ?? null);
    };

    const fetchData = async ({preserveActiveDevice = false, signal} = {}) => {
        try {
            if (!preserveActiveDevice) setIsLoading(true);
            const data = await fetchJson(`${API_URL}/api/data`, {}, signal);
            if (!data) return; // aborted

            // success
            failCountRef.current = 0;
            setHasLoadedOnce(true);
            setBannerError(null);
            setError(null);

            setDevices(data.devices);
            setAllActuators(data.all_actuators);

            const prev = preserveActiveDevice ? activeDeviceIdRef.current : null;
            setActiveDeviceId(pickDefaultActive(data.devices, prev));

            // inizializza / integra hysteresis
            setHysteresis(prevMap => {
                if (!prevMap || Object.keys(prevMap).length === 0) {
                    const init = {};
                    data.all_actuators.forEach(name => {
                        let found = null;
                        for (const d of data.devices) {
                            const a = d?.actuators?.[name];
                            if (a && typeof a === 'object' && a.thresholds) {
                                found = a.thresholds;
                                break;
                            }
                        }
                        init[name] = found || {on: 0.7, off: 0.3};
                    });
                    return init;
                }
                const copy = {...prevMap};
                data.all_actuators.forEach(name => {
                    if (!(name in copy)) copy[name] = {on: 0.7, off: 0.3};
                });
                return copy;
            });
        } catch (e) {
            if (e?.name === 'AbortError') return;
            failCountRef.current += 1;
            if (!hasLoadedOnce && (!devices || devices.length === 0)) {
                setError('Could not connect to the backend. Is the Python server running?');
            }
            if (!hasLoadedOnce ? failCountRef.current >= 1 : failCountRef.current >= 2) {
                setBannerError(e.message || 'Network error');
            }
            console.error('fetch data failed:', e);
        } finally {
            setIsLoading(false);
        }
    };

    const runFetch = ({preserveActiveDevice}) => {
        const ctrl = new AbortController();
        if (fetchCtrlRef.current) fetchCtrlRef.current.abort('superseded-by-new-fetch');
        fetchCtrlRef.current = ctrl;
        fetchData({preserveActiveDevice, signal: ctrl.signal});
    };

    useEffect(() => {
        runFetch({preserveActiveDevice: false});

        const timeInterval = setInterval(() => {
            setCurrentTime(new Date().toLocaleTimeString('it-IT'));
        }, 1000);

        const dataInterval = setInterval(() => {
            runFetch({preserveActiveDevice: true});
        }, 30000);

        return () => {
            if (fetchCtrlRef.current) fetchCtrlRef.current.abort('component-unmount');
            clearInterval(timeInterval);
            clearInterval(dataInterval);
        };
    }, []); // mount

    const handleApiAction = async (endpoint, body) => {
        setIsLoading(true);
        setError(null);
        setBannerError(null);
        const ctrl = new AbortController();
        try {
            const data = await fetchJson(`${API_URL}${endpoint}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body),
            }, ctrl.signal);
            if (!data) return false;
            setDevices(data.devices);
            setAllActuators(data.all_actuators);
            setActiveDeviceId(prev => pickDefaultActive(data.devices, prev));
            return true;
        } catch (e) {
            if (e?.name === 'AbortError') return false;
            const msg = `Operation failed: ${e.message}. Check backend connection.`;
            setError(msg);
            setBannerError(e.message);
            console.error(`Failed action at ${endpoint}:`, e);
            return false;
        } finally {
            setIsLoading(false);
        }
    };

    const handleRegisterDevice = (deviceData) => handleApiAction('/api/devices/add', deviceData);
    const handleUpdateDevice = (deviceId, deviceData) => handleApiAction('/api/devices/update', {deviceId, ...deviceData});
    const handleDisassociateDevice = (deviceId) => handleApiAction('/api/disassociate', {deviceId});
    const handleToggleActuator = (deviceId, actuatorName, isEnabled) => handleApiAction('/api/settings/actuator', {
        deviceId,
        actuatorName,
        isEnabled
    });

    async function handleSaveHysteresis(name, {on, off}) {
        const payload = [{actuatorName: name, on, off}]; // camelCase lato FE
        try {
            const res = await fetch(`${API_URL}/api/settings/hysteresis`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload),
            });
            const data = await res.json().catch(() => ({}));
            console.log('[UI] Hysteresis response:', res.status, data);
            // opzionale: ricarica dati
            await fetchData(true);
            return res.ok;
        } catch (e) {
            console.error('[UI] Hysteresis save failed:', e);
            return false;
        }
    }
    
    const activeDeviceData = useMemo(
        () => devices.find(d => d.device_id === activeDeviceId),
        [devices, activeDeviceId]
    );

    const renderContent = () => {
        if (isLoading && devices.length === 0) {
            return (
                <div className="p-8 text-white flex flex-col items-center justify-center h-full">
                    <Loader size={48} className="animate-spin text-indigo-400 mb-4"/>
                    Loading data...
                </div>
            );
        }
        if (error && devices.length === 0) {
            return (
                <div className="p-8 text-red-400 flex flex-col items-center justify-center h-full text-center">
                    <h2 className="text-2xl font-bold mb-4">Connection Error</h2>
                    <p>{error}</p>
                </div>
            );
        }
        switch (activePage) {
            case 'dashboard':
                return (
                    <Dashboard
                        deviceData={activeDeviceData}
                        currentTime={currentTime}
                        globalHysteresis={hysteresis}
                    />
                );
            case 'settings':
                return (
                    <SettingsPage
                        devices={devices}
                        allActuators={allActuators}
                        hysteresis={hysteresis}
                        onSaveHysteresis={handleSaveHysteresis}
                        onRegister={handleRegisterDevice}
                        onUpdate={handleUpdateDevice}
                        onDisassociate={handleDisassociateDevice}
                        onToggleActuator={handleToggleActuator}
                        isLoading={isLoading}
                    />
                );
            case 'history':
                return <div className="p-8 text-white">This section is under construction.</div>;
            default:
                return (
                    <Dashboard
                        deviceData={activeDeviceData}
                        currentTime={currentTime}
                        globalHysteresis={hysteresis}
                    />
                );
        }
    };

    return (
        <div className="flex h-screen bg-gray-800 font-sans">
            <Sidebar
                devices={devices}
                activeDeviceId={activeDeviceId}
                setActiveDeviceId={setActiveDeviceId}
                activePage={activePage}
                setActivePage={setActivePage}
            />
            <main className="flex-1 overflow-y-auto">
                {bannerError && devices.length === 0 && (
                    <div
                        className="m-3 p-3 rounded-md bg-red-900/40 text-red-300 text-sm flex justify-between items-center">
                        <span>{bannerError}</span>
                        <button className="px-2 py-1 bg-red-700/60 rounded"
                                onClick={() => setBannerError(null)}>Chiudi
                        </button>
                    </div>
                )}
                {renderContent()}
            </main>
        </div>
    );
}
