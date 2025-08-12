import React, { useState } from 'react';
import { Thermometer, Droplet, Wind, BrainCircuit, Activity, TrendingUp, Clock } from 'lucide-react';
import { getComfortStatus, formatValue, getPredictionTime } from '../utils/api';
import SensorCard from '../components/SensorCard';
import ExternalWeatherCard from '../components/ExternalWeatherCard';
import ActuatorCard from '../components/ActuatorCard';
import HistoryChart from '../components/HistoryChart';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const ComfortCircleWidget = ({ title, value, thresholds, higherIsBetter = true, unit = '%' }) => {
    const status = getComfortStatus(value, thresholds, higherIsBetter);
    return (
        <div className="bg-gray-800 rounded-full flex flex-col items-center justify-center w-36 h-36 ring-2 ring-gray-700">
            <span className="text-sm text-gray-400">{title}</span>
            <span className={`font-bold ${status.color} text-2xl`}>{status.text}</span>
            <span className="text-xs text-gray-500 mt-1">{formatValue(value, 0)}{unit}</span>
        </div>
    );
};

const PrimaryComfortCircleWidget = ({ title, value, thresholds }) => {
    const status = getComfortStatus(value, thresholds, true);
    return (
        <div className="bg-gray-800 rounded-full flex flex-col items-center justify-center w-48 h-48 ring-2 ring-indigo-500">
            <span className="text-sm text-gray-400">{title}</span>
            <span className={`font-bold ${status.color} text-4xl`}>{status.text}</span>
            <span className="text-xs text-gray-500 mt-1">{formatValue(value, 0)}%</span>
        </div>
    );
};

const PredictionCarousel = ({ predictions }) => {
    const [currentIndex, setCurrentIndex] = useState(0);
    const keys = Object.keys(predictions || {});
    if (!keys || keys.length === 0) return <p className="text-xs text-gray-500 text-center py-8">No predictions available.</p>;
    const horizon = keys[currentIndex];
    const result = predictions[horizon];
    const nav = (dir) => setCurrentIndex((i) => (dir === 'next' ? (i + 1) % keys.length : (i - 1 + keys.length) % keys.length));
    return (
        <div className="flex items-center gap-2" aria-live="polite">
            {keys.length > 1 && <button onClick={() => nav('prev')} className="text-gray-400 hover:text-white transition-colors">‹</button>}
            <div className="border border-gray-600 p-3 rounded-lg flex-grow">
                <div className="flex justify-between items-center mb-2"><p className="text-sm text-indigo-400 font-bold">In {horizon}</p></div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm text-gray-300">
                    <span className="flex items-center"><Thermometer className="inline mr-2 text-red-400" size={16} />{formatValue(result?.temperature, 1)}°C</span>
                    <span className="flex items-center"><Droplet className="inline mr-2 text-blue-400" size={16} />{formatValue(result?.humidity)}%</span>
                    <span className="flex items-center"><Wind className="inline mr-2 text-gray-400" size={16} />{formatValue(result?.co2)}ppm</span>
                    <span className="flex items-center"><BrainCircuit className="inline mr-2 text-purple-400" size={16} />{formatValue(result?.voc)}ppb</span>
                </div>
                <div className="mt-3 border-t border-gray-700 pt-2 grid grid-cols-3 gap-2 text-xs">
                    <div className="text-center"><span className="text-gray-400 block">Comfort</span><span className="font-semibold text-white">{formatValue(result?.globalComfort)}%</span></div>
                    <div className="text-center"><span className="text-gray-400 block">Heat Idx</span><span className="font-semibold text-white">{formatValue(result?.heatIndex, 1)}°</span></div>
                    <div className="text-center"><span className="text-gray-400 block">IAQ Idx</span><span className="font-semibold text-white">{formatValue(result?.iaqIndex)}%</span></div>
                </div>
            </div>
            {keys.length > 1 && <button onClick={() => nav('next')} className="text-gray-400 hover:text-white transition-colors">›</button>}
        </div>
    );
};

const PredictionCard = ({ predictions }) => (
    <div className="bg-gray-900 p-4 rounded-lg h-full">
        <h3 className="text-white font-semibold mb-3 flex items-center"><TrendingUp size={20} className="mr-2 text-indigo-400" />What will happen?</h3>
        <PredictionCarousel predictions={predictions} />
    </div>
);

const SuggestionCard = ({ suggestions }) => (
    <div className="bg-gray-900 p-4 rounded-lg">
        <h3 className="text-white font-semibold mb-3 flex items-center">What you can do?</h3>
        {suggestions && suggestions.length > 0 ? (
            <div className="space-y-4">
                {suggestions.map(({ action, predictions = {} }, idx) => (
                    <div key={idx} className="bg-gray-800 p-3 rounded-lg">
                        <p className="text-white font-medium mb-3">{action}</p>
                        <PredictionCarousel predictions={predictions} />
                    </div>
                ))}
            </div>
        ) : <p className="text-xs text-gray-500 text-center py-4">No suggestions available.</p>}
    </div>
);

const Dashboard = ({ deviceData, currentTime, globalHysteresis }) => {
    const [chartView, setChartView] = useState('sensors');
    if (!deviceData || !deviceData.current || Object.keys(deviceData.current).length === 0) {
        return (
            <div className="p-8 text-white flex flex-col items-center justify-center h-full">
                <h2 className="text-2xl font-bold">Select a device</h2>
                <p className="text-gray-400">Please select a device from the sidebar to view its dashboard.</p>
            </div>
        );
    }
    const ChartButton = ({ view, label }) => (
        <button onClick={() => setChartView(view)} className={`px-3 py-1 text-xs rounded-md transition-colors ${chartView === view ? 'bg-indigo-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}>{label}</button>
    );

    return (
        <div className="p-4 md:p-6 space-y-6">
            <div className="flex justify-between items-center">
                <h2 className="text-3xl font-bold text-white">Dashboard: {deviceData.room_name}</h2>
                <div className="flex items-center text-lg text-gray-300 bg-gray-900/50 px-4 py-2 rounded-lg">
                    <Clock size={20} className="mr-2" />{currentTime}
                </div>
            </div>

            <div className="flex justify-center items-center gap-8 py-4">
                <ComfortCircleWidget title="Heat Index" value={deviceData.current.heatIndex} thresholds={{ good: 21, medium: 25 }} higherIsBetter={false} unit="°" />
                <PrimaryComfortCircleWidget title="Global Comfort" value={deviceData.current.globalComfort} thresholds={{ good: 80, medium: 60 }} />
                <ComfortCircleWidget title="IAQ Index" value={deviceData.current.iaqIndex} thresholds={{ good: 80, medium: 60 }} />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="flex flex-col space-y-6">
                    <ExternalWeatherCard weather={deviceData.externalWeather} />
                    <div className="bg-gray-900 p-4 rounded-lg">
                        <h3 className="text-white font-semibold mb-3 flex items-center"><Activity size={20} className="mr-2 text-gray-400" />How is it now?</h3>
                        <div className="grid grid-cols-2 gap-4">
                            <SensorCard title="Temperature" icon={<Thermometer className="text-red-400" size={24} />} value={deviceData.current.temperature} unit="°C" decimals={1} />
                            <SensorCard title="Humidity" icon={<Droplet className="text-blue-400" size={24} />} value={deviceData.current.humidity} unit="%" />
                            <SensorCard title="CO2" icon={<Wind className="text-gray-400" size={24} />} value={deviceData.current.co2} unit="ppm" />
                            <SensorCard title="VOC" icon={<BrainCircuit className="text-purple-400" size={24} />} value={deviceData.current.voc} unit="ppb" />
                        </div>
                    </div>

                    <div className="bg-gray-900 p-4 rounded-lg">
                        <ActuatorCard
                            actuators={deviceData.actuators}
                            available_actuators={deviceData.available_actuators}
                            predAt={getPredictionTime(deviceData)}
                            globalHysteresis={globalHysteresis}
                        />
                    </div>
                </div>

                <div className="flex flex-col space-y-6">
                    <PredictionCard predictions={deviceData.predictions} />
                    <SuggestionCard suggestions={deviceData.suggestions} />
                </div>
            </div>

            <div className="bg-gray-900 p-4 rounded-lg mt-6">
                <div className="flex flex-wrap gap-2 justify-between items-center mb-4">
                    <h3 className="text-white font-semibold text-base">What happened?</h3>
                    <div className="flex flex-wrap gap-1 bg-gray-800 p-1 rounded-md">
                        <ChartButton view="sensors" label="Sensors" />
                        <ChartButton view="globalComfort" label="Comfort" />
                        <ChartButton view="heatIndex" label="Heat Idx" />
                        <ChartButton view="iaqIndex" label="IAQ Idx" />
                    </div>
                </div>
                <div className="h-80">
                    <HistoryChart data={deviceData.history || []} view={chartView} />
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
