import React from 'react';
import { Sun, Cloud, CloudRain, Thermometer, Droplet, Wind } from 'lucide-react';
import { formatValue } from '../utils/api';

const ExternalWeatherCard = ({ weather }) => {
    if (!weather) return <div className="bg-gray-900 p-4 rounded-lg text-center"><p className="text-gray-400">Weather data not available.</p></div>;
    const WeatherIcon = () => {
        switch (weather.icon) {
            case 'sun': case 'clear': return <Sun className="w-16 h-16 text-yellow-400" />;
            case 'clouds': return <Cloud className="w-16 h-16 text-gray-400" />;
            case 'rain': return <CloudRain className="w-16 h-16 text-blue-400" />;
            default: return <Cloud className="w-16 h-16 text-yellow-400" />;
        }
    };
    return (
        <div className="bg-gray-900 p-4 rounded-lg">
            <div className="flex items-center justify-around text-center">
                <div>
                    <p className="text-lg font-bold text-indigo-400">{weather.location}</p>
                    <WeatherIcon />
                </div>
                <div className="space-y-2 text-left">
                    <p className="text-2xl font-bold text-white flex items-center"><Thermometer size={20} className="mr-2 text-red-400" /> {formatValue(weather.temperature, 1)}°C</p>
                    <p className="text-lg text-gray-300 flex items-center"><Droplet size={16} className="mr-2 text-blue-400" /> {formatValue(weather.humidity)}%</p>
                    <p className="text-lg text-gray-300 flex items-center"><Wind size={16} className="mr-2 text-gray-400" /> {formatValue(weather.wind, 1)} km/h</p>
                </div>
            </div>
        </div>
    );
};

export default ExternalWeatherCard;
