import React, { useState, useEffect } from 'react';
import { Settings as SettingsIcon, ToggleRight, PlusCircle, Edit, Save, XCircle, Unlink } from 'lucide-react';

const DeviceEditor = ({ device, allActuators, onSave, onCancel, onToggleActuator, onDisassociate, isLoading }) => {
    const [editData, setEditData] = useState(device);
    useEffect(() => setEditData(device), [device]);

    const handleSave = () => {
        onSave(device.device_id, {
            roomName: editData.room_name,
            locationName: editData.location_name,
            lat: editData.lat,
            lng: editData.lng
        });
    };

    return (
        <div className="bg-gray-800 p-4 rounded-md">
            <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                    <label className="text-xs text-gray-400">Room Name</label>
                    <input
                        type="text"
                        value={editData.room_name || ''}
                        onChange={(e) => setEditData({ ...editData, room_name: e.target.value })}
                        className="w-full p-2 mt-1 rounded-md bg-gray-700 text-white border border-gray-600"
                    />
                </div>
                <div>
                    <label className="text-xs text-gray-400">Location Name</label>
                    <input
                        type="text"
                        value={editData.location_name || ''}
                        onChange={(e) => setEditData({ ...editData, location_name: e.target.value })}
                        className="w-full p-2 mt-1 rounded-md bg-gray-700 text-white border border-gray-600"
                    />
                </div>
                <div>
                    <label className="text-xs text-gray-400">Latitude</label>
                    <input
                        type="number"
                        value={editData.lat || ''}
                        onChange={(e) => setEditData({ ...editData, lat: e.target.value })}
                        className="w-full p-2 mt-1 rounded-md bg-gray-700 text-white border border-gray-600"
                    />
                </div>
                <div>
                    <label className="text-xs text-gray-400">Longitude</label>
                    <input
                        type="number"
                        value={editData.lng || ''}
                        onChange={(e) => setEditData({ ...editData, lng: e.target.value })}
                        className="w-full p-2 mt-1 rounded-md bg-gray-700 text-white border border-gray-600"
                    />
                </div>
            </div>

            <div className="mb-4">
                <h4 className="text-sm font-semibold text-gray-400 mb-2">Available Actuators</h4>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                    {allActuators.map((actuatorName) => (
                        <label
                            key={actuatorName}
                            className="flex items-center space-x-2 bg-gray-700 p-2 rounded-md cursor-pointer hover:bg-gray-600"
                        >
                            <input
                                type="checkbox"
                                className="form-checkbox h-5 w-5 bg-gray-800 border-gray-600 rounded text-indigo-500 focus:ring-indigo-500"
                                checked={device.available_actuators?.[actuatorName] || false}
                                onChange={(e) => onToggleActuator(device.device_id, actuatorName, e.target.checked)}
                                disabled={isLoading}
                            />
                            <span className="text-gray-300 text-sm">{actuatorName}</span>
                        </label>
                    ))}
                </div>
            </div>

            <div className="flex justify-between items-center">
                <button
                    onClick={() => onDisassociate(device.device_id)}
                    disabled={isLoading || !device.room_name}
                    className="flex items-center text-sm px-3 py-1 bg-red-600 hover:bg-red-500 rounded transition-colors disabled:bg-gray-500 disabled:cursor-not-allowed"
                >
                    <Unlink size={14} className="mr-1.5" /> Unassign
                </button>
                <div className="flex gap-2">
                    <button
                        type="button"
                        onClick={onCancel}
                        className="px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded-md flex items-center transition-colors"
                    >
                        <XCircle size={16} className="mr-1.5" /> Cancel
                    </button>
                    <button
                        type="button"
                        onClick={handleSave}
                        disabled={isLoading}
                        className="px-4 py-2 bg-green-600 hover:bg-green-500 rounded-md flex items-center transition-colors disabled:opacity-70"
                    >
                        <Save size={16} className="mr-1.5" /> Save
                    </button>
                </div>
            </div>
        </div>
    );
};

const HysteresisRow = ({ name, value, onChange }) => {
    const [onVal, setOnVal] = useState(value?.on ?? '');
    const [offVal, setOffVal] = useState(value?.off ?? '');
    const [saving, setSaving] = useState(false);
    const [savedOk, setSavedOk] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
        setOnVal(value?.on ?? '');
        setOffVal(value?.off ?? '');
        setSavedOk(false);
        setError('');
    }, [value?.on, value?.off]);

    const parsedOn = parseFloat(onVal);
    const parsedOff = parseFloat(offVal);
    const valid =
        !Number.isNaN(parsedOn) &&
        !Number.isNaN(parsedOff) &&
        parsedOff >= 0 &&
        parsedOn <= 1 &&
        parsedOff < parsedOn;

    const handleSave = async () => {
        if (!valid) return;
        try {
            setSaving(true);
            setSavedOk(false);
            setError('');
            // mantengo la firma: onChange({ on, off })
            const maybePromise = onChange({ on: parsedOn, off: parsedOff });
            const ok = typeof maybePromise?.then === 'function' ? await maybePromise : true;
            setSavedOk(ok !== false);
            if (ok === false) setError('Save failed');
        } catch (e) {
            setError('Save failed');
        } finally {
            setSaving(false);
        }
    };

    return (
        <div className="grid grid-cols-12 gap-3 items-center bg-gray-800 p-3 rounded-md">
            <div className="col-span-4 text-gray-200">{name}</div>

            <div className="col-span-3">
                <label className="text-xs text-gray-400">ON</label>
                <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    value={onVal}
                    onChange={(e) => setOnVal(e.target.value)}
                    className="w-full p-2 mt-1 rounded-md bg-gray-700 text-white border border-gray-600"
                />
            </div>

            <div className="col-span-3">
                <label className="text-xs text-gray-400">OFF</label>
                <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    value={offVal}
                    onChange={(e) => setOffVal(e.target.value)}
                    className="w-full p-2 mt-1 rounded-md bg-gray-700 text-white border border-gray-600"
                />
            </div>

            <div className="col-span-2 flex items-center justify-end gap-2">
                {savedOk && <span className="text-xs text-green-400">Saved ✓</span>}
                {error && <span className="text-xs text-red-300">{error}</span>}
                <button
                    type="button"
                    disabled={!valid || saving}
                    onClick={handleSave}
                    className={`px-3 py-2 rounded-md ${valid && !saving ? 'bg-indigo-600 hover:bg-indigo-500' : 'bg-gray-600 cursor-not-allowed'} transition-colors`}
                    title={!valid ? 'OFF < ON, entrambe tra 0 e 1' : 'Salva'}
                >
                    {saving ? 'Saving…' : 'Save'}
                </button>
            </div>

            {!valid && (
                <div className="col-span-12 text-xs text-red-300">
                    OFF deve essere &lt; ON, entrambe tra 0 e 1.
                </div>
            )}
        </div>
    );
};

const SettingsPage = ({
                          devices,
                          allActuators,
                          hysteresis,
                          onSaveHysteresis,      // (name, {on, off}) => Promise<boolean>|boolean
                          onRegister,
                          onUpdate,
                          onDisassociate,
                          onToggleActuator,
                          isLoading
                      }) => {
    const [editingDeviceId, setEditingDeviceId] = useState(null);
    const [newDeviceData, setNewDeviceData] = useState({ deviceId: '', locationName: '', lat: '', lng: '' });

    const handleRegisterClick = () => {
        if (!newDeviceData.deviceId.trim()) {
            alert('Please provide a Device ID.');
            return;
        }
        onRegister(newDeviceData);
        setNewDeviceData({ deviceId: '', locationName: '', lat: '', lng: '' });
    };

    return (
        <div className="p-8 space-y-8 text-white">
            <h1 className="text-3xl font-bold mb-6">Device & Room Settings</h1>

            {/* Devices */}
            <div className="bg-gray-900 p-6 rounded-lg">
                <h2 className="text-xl font-semibold mb-4 flex items-center">
                    <SettingsIcon size={20} className="mr-3 text-indigo-400" />
                    All Devices
                </h2>
                <div className="space-y-4">
                    {devices.length > 0 ? (
                        devices.map((device) =>
                            editingDeviceId === device.device_id ? (
                                <DeviceEditor
                                    key={device.device_id}
                                    device={device}
                                    allActuators={allActuators}
                                    onSave={(id, data) => {
                                        onUpdate(id, data);
                                        setEditingDeviceId(null);
                                    }}
                                    onCancel={() => setEditingDeviceId(null)}
                                    onToggleActuator={onToggleActuator}
                                    onDisassociate={onDisassociate}
                                    isLoading={isLoading}
                                />
                            ) : (
                                <div
                                    key={device.device_id}
                                    className="flex justify-between items-center bg-gray-800 p-4 rounded-md"
                                >
                                    <div>
                                        <h3 className="font-bold text-lg text-indigo-300">
                                            {device.room_name || '(Unassigned)'}
                                        </h3>
                                        <p className="text-xs text-gray-500 font-mono">{device.device_id}</p>
                                    </div>
                                    <button
                                        type="button"
                                        onClick={() => setEditingDeviceId(device.device_id)}
                                        className="px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded-md flex items-center transition-colors"
                                    >
                                        <Edit size={16} className="mr-1.5" /> Edit
                                    </button>
                                </div>
                            )
                        )
                    ) : (
                        <p className="text-gray-400 text-sm">No devices registered yet.</p>
                    )}
                </div>
            </div>

            {/* Hysteresis */}
            <div className="bg-gray-900 p-6 rounded-lg">
                <h2 className="text-xl font-semibold mb-4 flex items-center">
                    <ToggleRight size={18} className="mr-2 text-gray-400" />
                    Hysteresis (global per attuatore)
                </h2>
                {allActuators?.length ? (
                    <div className="space-y-3">
                        {allActuators.map((name) => (
                            <HysteresisRow
                                key={name}
                                name={name}
                                value={hysteresis?.[name]}
                                onChange={(val) => onSaveHysteresis(name, val)}
                            />
                        ))}
                    </div>
                ) : (
                    <p className="text-gray-400 text-sm">No actuators found.</p>
                )}
            </div>

            {/* Register Device */}
            <div className="bg-gray-900 p-6 rounded-lg">
                <h2 className="text-xl font-semibold mb-4 flex items-center">
                    <PlusCircle size={18} className="mr-2" /> Register New Device
                </h2>
                <div className="grid grid-cols-2 gap-4 mb-4">
                    <input
                        type="text"
                        value={newDeviceData.deviceId}
                        onChange={(e) => setNewDeviceData({ ...newDeviceData, deviceId: e.target.value })}
                        placeholder="New Device ID"
                        className="col-span-2 p-2 rounded-md bg-gray-700 text-white border border-gray-600"
                    />
                    <input
                        type="text"
                        value={newDeviceData.locationName}
                        onChange={(e) => setNewDeviceData({ ...newDeviceData, locationName: e.target.value })}
                        placeholder="Location Name (e.g., Milan)"
                        className="p-2 rounded-md bg-gray-700 text-white border border-gray-600"
                    />
                    <input
                        type="number"
                        value={newDeviceData.lat}
                        onChange={(e) => setNewDeviceData({ ...newDeviceData, lat: e.target.value })}
                        placeholder="Latitude"
                        className="p-2 rounded-md bg-gray-700 text-white border border-gray-600"
                    />
                    <input
                        type="number"
                        value={newDeviceData.lng}
                        onChange={(e) => setNewDeviceData({ ...newDeviceData, lng: e.target.value })}
                        placeholder="Longitude"
                        className="p-2 rounded-md bg-gray-700 text-white border border-gray-600"
                    />
                </div>
                <button
                    type="button"
                    onClick={handleRegisterClick}
                    disabled={isLoading}
                    className="w-full px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-md flex items-center justify-center transition-colors disabled:opacity-70"
                >
                    Register Device
                </button>
            </div>
        </div>
    );
};

export default SettingsPage;
