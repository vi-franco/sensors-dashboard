import React from 'react';
import { Fan, LayoutDashboard, History, Settings } from 'lucide-react';

const Sidebar = ({ devices, activeDeviceId, setActiveDeviceId, activePage, setActivePage }) => {
    const NavLink = ({ page, icon, children }) => (
        <button
            onClick={() => setActivePage(page)}
            className={`flex items-center w-full text-left px-4 py-2.5 rounded-lg transition-colors duration-200 ${activePage === page ? 'bg-indigo-600 text-white' : 'text-gray-300 hover:bg-gray-700'}`}
        >
            {icon}<span className="ml-3">{children}</span>
        </button>
    );
    const assignedDevices = devices.filter(d => d.room_name);
    return (
        <div className="w-64 bg-gray-900 text-white h-screen flex flex-col p-4 shrink-0">
            <div className="text-2xl font-bold text-white mb-8 flex items-center">
                <Fan className="mr-2 animate-spin-slow" />SensorHub
            </div>
            <nav className="flex-grow">
                <NavLink page="dashboard" icon={<LayoutDashboard size={20} />}>Dashboard</NavLink>
                <div className="mt-4 pl-4">
                    <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Assigned Devices</h3>
                    {assignedDevices.length > 0 ? assignedDevices.map(device => (
                        <button
                            key={device.device_id}
                            onClick={() => { setActivePage('dashboard'); setActiveDeviceId(device.device_id); }}
                            className={`block w-full text-left text-sm py-2 px-3 rounded-md transition-colors duration-200 ${activePage === 'dashboard' && activeDeviceId === device.device_id ? 'bg-gray-700 text-white' : 'text-gray-400 hover:bg-gray-700'}`}
                        >
                            {device.room_name}
                        </button>
                    )) : <p className="text-xs text-gray-500 px-3">No devices assigned to a room.</p>}
                </div>
                <div className="mt-6 border-t border-gray-700 pt-6">
                    <NavLink page="history" icon={<History size={20} />}>History</NavLink>
                    <NavLink page="settings" icon={<Settings size={20} />}>Settings</NavLink>
                </div>
            </nav>
        </div>
    );
};

export default Sidebar;
