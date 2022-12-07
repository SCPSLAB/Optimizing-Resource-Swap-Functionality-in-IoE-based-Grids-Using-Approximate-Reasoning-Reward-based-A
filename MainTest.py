#!/usr/bin/env python
# coding: utf-8



import numpy as np
import os
import random
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.models import load_model
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam


class SmartHome:  
    def __init__(self, SmartHomeId, DHWT= None, CWT= None, elect_stg = None, DHW_AC_app= None, CWT_AC_app= None, save_memory = True):

        self.SmartHome_type = None
        self.MicroArea_Type= None
        self.PV_capacity = None
        self.SmartHomeId = SmartHomeId
        self.DHWT= dhw_stg
        self.CWT= cooling_stg
        self.elect_stg = electrical_stg
        self.DHW_AC_app= dhw_heating_app
        self.CWT_AC_app= cooling_app
        self.obs_space = None
        self.action_space = None
        self.time_step = 0
        self.sim_results = {}
        self.save_memory = save_memory
        
        if self.DHWT is not None:
            self.dhw_stg.reset()
        if self.CWT is not None:
            self.cooling_stg.reset()
        if self.elect_stg is not None:
            self.electrical_stg.reset()
        if self.DHW_AC_app is not None:
            self.dhw_heating_app.reset()
        if self.CWT_AC_app is not None:
            self.cooling_app.reset()
            
        self._electric_usage_CWT= 0.0
        self._electric_usage_DHWT= 0.0
        
        self.cooling_demand_SmartHome = []
        self.dhw_demand_SmartHome = []
        self.electric_usage_appliances = []
        self.electric_gen= []
           
        self.electric_usage_cooling = []
        self.electric_usage_CWT= []
        self.electric_usage_dhw = []
        self.electric_usage_DHWT= []
        
        self.net_electric_usage = []
        self.net_electric_usage_no_stg = []
        self.net_electric_usage_no_pv_no_stg = []
        
        self.cooling_app_to_SmartHome = []
        self.cooling_stg_to_SmartHome = []
        self.cooling_app_to_stg = []
        self.cooling_stg_stateofcharge= []

        self.dhw_heating_app_to_SmartHome = []
        self.dhw_stg_to_SmartHome = []
        self.dhw_heating_app_to_stg = []
        self.dhw_stg_stateofcharge= []
        
        self.electrical_stg_electric_usage = []
        self.electrical_stg_stateofcharge= []
        
    def set_state_space(self, high_state, low_state):
        self.obs_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)
    
    def set_action_space(self, max_action, min_action):
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)
        
    def set_stg_electrical(self, action):


        electrical_energy_balance = self.electrical_stg.charge(action*self.electrical_stg.capacity)
        
        if self.save_memory == False:
            self.electrical_stg_electric_usage.append(electrical_energy_balance)
            self.electrical_stg_soc.append(self.electrical_stg._soc)
        
        self.electrical_stg.time_step += 1
        
        return electrical_energy_balance
    

    def set_stg_heating(self, action):

        
        heat_power_w_avail = self.dhw_heating_app.get_max_heating_power_w() - self.sim_results['dhw_demand'][self.time_step]
        
        heating_energy_balance = self.dhw_stg.charge(max(-self.sim_results['dhw_demand'][self.time_step], min(heat_power_w_avail, action*self.dhw_stg.capacity)))
        
        if self.save_memory == False:
            self.dhw_heating_app_to_stg.append(max(0, heating_energy_balance))
            self.dhw_stg_to_SmartHome.append(-min(0, heating_energy_balance))
            self.dhw_heating_app_to_SmartHome.append(self.sim_results['dhw_demand'][self.time_step] + min(0, heating_energy_balance))
            self.dhw_stg_soc.append(self.dhw_stg._soc)
        
        heating_energy_balance = max(0, heating_energy_balance + self.sim_results['dhw_demand'][self.time_step])
        
        elec_demand_heating = self.dhw_heating_app.set_total_electric_usage_heating(heat_supply = heating_energy_balance)
        
        self._electric_usage_DHWT= elec_demand_heating - self.dhw_heating_app.get_electric_usage_heating(heat_supply = self.sim_results['dhw_demand'][self.time_step])
        
        if self.save_memory == False:
            self.electric_usage_dhw.append(elec_demand_heating)
            self.electric_usage_dhw_stg.append(self._electric_usage_dhw_stg)
        
        self.dhw_heating_app.time_step += 1
        
        return elec_demand_heating
    
        
    def set_stg_cooling(self, action):

    
        cooling_power_w_avail = self.cooling_app.get_max_cooling_power_w() - self.sim_results['cooling_demand'][self.time_step]
        
        if isinstance(action, list):
            action = action[0]
        cooling_energy_balance = self.cooling_stg.charge(max(-self.sim_results['cooling_demand'][self.time_step], min(cooling_power_w_avail, action*self.cooling_stg.capacity))) 
        
        if self.save_memory == False:
            self.cooling_app_to_stg.append(max(0, cooling_energy_balance))
            self.cooling_stg_to_SmartHome.append(-min(0, cooling_energy_balance))
            self.cooling_app_to_SmartHome.append(self.sim_results['cooling_demand'][self.time_step] + min(0, cooling_energy_balance))
            self.cooling_stg_soc.append(self.cooling_stg._soc)
        
        cooling_energy_balance = max(0, cooling_energy_balance + self.sim_results['cooling_demand'][self.time_step])
        
        elec_demand_cooling = self.cooling_app.set_total_electric_usage_cooling(cooling_supply = cooling_energy_balance)
        
        self._electric_usage_CWT= elec_demand_cooling - self.cooling_app.get_electric_usage_cooling(cooling_supply = self.sim_results['cooling_demand'][self.time_step])
        
        if self.save_memory == False:
            self.electric_usage_cooling.append(np.float32(elec_demand_cooling))
            self.electric_usage_cooling_stg.append(np.float32(self._electric_usage_cooling_stg))
            
        self.cooling_app.time_step += 1

        return elec_demand_cooling
    

    def get_non_shiftable_load(self):
        return self.sim_results['non_shiftable_load'][self.time_step]
    
    def get_solar_power_w(self):
        return self.sim_results['solar_gen'][self.time_step]
    
    def get_dhw_electric_demand(self):
        return self.dhw_heating_app._electrical_usage_heating
        
    def get_cooling_electric_demand(self):
        return self.cooling_app._electrical_usage_cooling
    
    def reset(self):
        
        self.current_net_electricity_demand = self.sim_results['non_shiftable_load'][self.time_step] - self.sim_results['solar_gen'][self.time_step]
        
        if self.DHWT is not None:
            self.dhw_stg.reset()
        if self.CWT is not None:
            self.cooling_stg.reset()
        if self.elect_stg is not None:
            self.electrical_stg.reset()
        if self.DHW_AC_app is not None:
            self.dhw_heating_app.reset()
            self.current_net_electricity_demand += self.dhw_heating_app.get_electric_usage_heating(self.sim_results['dhw_demand'][self.time_step]) 
        if self.CWT_AC_app is not None:
            self.cooling_app.reset()
            self.current_net_electricity_demand += self.cooling_app.get_electric_usage_cooling(self.sim_results['cooling_demand'][self.time_step])
            
        self._electric_usage_CWT= 0.0
        self._electric_usage_DHWT= 0.0
        self.cooling_demand_SmartHome = []
        self.dhw_demand_SmartHome = []
        self.electric_usage_appliances = []
        self.electric_gen= []
        self.electric_usage_cooling = []
        self.electric_usage_CWT= []
        self.electric_usage_dhw = []
        self.electric_usage_DHWT= [] 
        self.net_electric_usage = []
        self.net_electric_usage_no_stg = []
        self.net_electric_usage_no_pv_no_stg = []
        self.cooling_app_to_SmartHome = []
        self.cooling_stg_to_SmartHome = []
        self.cooling_app_to_stg = []
        self.cooling_stg_stateofcharge= []
        self.dhw_heating_app_to_SmartHome = []
        self.dhw_stg_to_SmartHome = []
        self.dhw_heating_app_to_stg = []
        self.dhw_stg_stateofcharge= []
        self.electrical_stg_electric_usage = []
        self.electrical_stg_stateofcharge= []
        
    def terminate(self):
        
        if self.DHWT is not None:
            self.dhw_stg.terminate()
        if self.CWT is not None:
            self.cooling_stg.terminate()
        if self.elect_stg is not None:
            self.electrical_stg.terminate()
        if self.DHW_AC_app is not None:
            self.dhw_heating_app.terminate()
        if self.CWT_AC_app is not None:
            self.cooling_app.terminate()
            
        if self.save_memory == False:
            
            self.cooling_demand_SmartHome = np.array(self.sim_results['cooling_demand'][:self.time_step])
            self.dhw_demand_SmartHome = np.array(self.sim_results['dhw_demand'][:self.time_step])
            self.electric_usage_appliances = np.array(self.sim_results['non_shiftable_load'][:self.time_step])
            self.electric_gen= np.array(self.sim_results['solar_gen'][:self.time_step])
            
            elec_usage_dhw = 0
            elec_usage_DHWT= 0
            if self.dhw_heating_app.time_step == self.time_step and self.DHW_AC_app is not None:
                elec_usage_dhw = np.array(self.electric_usage_dhw)
                elec_usage_DHWT= np.array(self.electric_usage_dhw_stg)
                
            elec_usage_cooling = 0
            elec_usage_CWT= 0
            if self.cooling_app.time_step == self.time_step and self.CWT_AC_app is not None:
                elec_usage_cooling = np.array(self.electric_usage_cooling)
                elec_usage_CWT= np.array(self.electric_usage_cooling_stg)
                
            self.net_electric_usage = np.array(self.electric_usage_appliances) + elec_usage_cooling + elec_usage_dhw - np.array(self.electric_generation) 
            self.net_electric_usage_no_stg = np.array(self.electric_usage_appliances) + (elec_usage_cooling - elec_usage_cooling_stg) + (elec_usage_dhw - elec_usage_dhw_stg) - np.array(self.electric_generation)
            self.net_electric_usage_no_pv_no_stg = np.array(self.net_electric_usage_no_stg) + np.array(self.electric_generation)
            self.cooling_demand_SmartHome = np.array(self.cooling_demand_SmartHome)
            self.dhw_demand_SmartHome = np.array(self.dhw_demand_SmartHome)
            self.electric_usage_appliances = np.array(self.electric_usage_appliances)
            self.electric_gen= np.array(self.electric_generation)
            self.electric_usage_cooling = np.array(self.electric_usage_cooling)
            self.electric_usage_CWT= np.array(self.electric_usage_cooling_stg)
            self.electric_usage_dhw = np.array(self.electric_usage_dhw)
            self.electric_usage_DHWT= np.array(self.electric_usage_dhw_stg)
            self.net_electric_usage = np.array(self.net_electric_usage)
            self.net_electric_usage_no_stg = np.array(self.net_electric_usage_no_stg)
            self.net_electric_usage_no_pv_no_stg = np.array(self.net_electric_usage_no_pv_no_stg)
            self.cooling_app_to_SmartHome = np.array(self.cooling_app_to_SmartHome)
            self.cooling_stg_to_SmartHome = np.array(self.cooling_stg_to_SmartHome)
            self.cooling_app_to_stg = np.array(self.cooling_app_to_stg)
            self.cooling_stg_stateofcharge= np.array(self.cooling_stg_soc)
            self.dhw_heating_app_to_SmartHome = np.array(self.dhw_heating_app_to_SmartHome)
            self.dhw_stg_to_SmartHome = np.array(self.dhw_stg_to_SmartHome)
            self.dhw_heating_app_to_stg = np.array(self.dhw_heating_app_to_stg)
            self.dhw_stg_stateofcharge= np.array(self.dhw_stg_soc)
            self.electrical_stg_electric_usage = np.array(self.electrical_stg_electric_usage)
            self.electrical_stg_stateofcharge= np.array(self.electrical_stg_soc)
            
class Building:  
    def __init__(self, buildingId, dhw_storage = None, cooling_storage = None, electrical_storage = None, dhw_heating_device = None, cooling_device = None, save_memory = True):

        self.building_type = None
        self.climate_zone = None
        self.solar_power_capacity = None
        self.buildingId = buildingId
        self.dhw_storage = dhw_storage
        self.cooling_storage = cooling_storage
        self.electrical_storage = electrical_storage
        self.dhw_heating_device = dhw_heating_device
        self.cooling_device = cooling_device
        self.observation_space = None
        self.action_space = None
        self.time_step = 0
        self.sim_results = {}
        self.save_memory = save_memory
    
        if self.dhw_storage is not None:
            self.dhw_storage.reset()
        if self.cooling_storage is not None:
            self.cooling_storage.reset()
        if self.electrical_storage is not None:
            self.electrical_storage.reset()
        if self.dhw_heating_device is not None:
            self.dhw_heating_device.reset()
        if self.cooling_device is not None:
            self.cooling_device.reset()
            
        self._electric_consumption_cooling_storage = 0.0
        self._electric_consumption_dhw_storage = 0.0
        self.cooling_demand_building = []
        self.dhw_demand_building = []
        self.electric_consumption_appliances = []
        self.electric_generation = []
        self.electric_consumption_cooling = []
        self.electric_consumption_cooling_storage = []
        self.electric_consumption_dhw = []
        self.electric_consumption_dhw_storage = []
        self.net_electric_consumption = []
        self.net_electric_consumption_no_storage = []
        self.net_electric_consumption_no_pv_no_storage = []
        self.cooling_device_to_building = []
        self.cooling_storage_to_building = []
        self.cooling_device_to_storage = []
        self.cooling_storage_soc = []
        self.dhw_heating_device_to_building = []
        self.dhw_storage_to_building = []
        self.dhw_heating_device_to_storage = []
        self.dhw_storage_soc = []
        self.electrical_storage_electric_consumption = []
        self.electrical_storage_soc = []
        
    def set_state_space(self, high_state, low_state):
        self.observation_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)
    
    def set_action_space(self, max_action, min_action):
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)
        
    def set_storage_electrical(self, action):


        electrical_energy_balance = self.electrical_storage.charge(action*self.electrical_storage.capacity)
        
        if self.save_memory == False:
            self.electrical_storage_electric_consumption.append(electrical_energy_balance)
            self.electrical_storage_soc.append(self.electrical_storage._soc)
        
        self.electrical_storage.time_step += 1
        
        return electrical_energy_balance
    

    def set_storage_heating(self, action):

        heat_power_avail = self.dhw_heating_device.get_max_heating_power() - self.sim_results['dhw_demand'][self.time_step]
        
        heating_energy_balance = self.dhw_storage.charge(max(-self.sim_results['dhw_demand'][self.time_step], min(heat_power_avail, action*self.dhw_storage.capacity)))
        
        if self.save_memory == False:
            self.dhw_heating_device_to_storage.append(max(0, heating_energy_balance))
            self.dhw_storage_to_building.append(-min(0, heating_energy_balance))
            self.dhw_heating_device_to_building.append(self.sim_results['dhw_demand'][self.time_step] + min(0, heating_energy_balance))
            self.dhw_storage_soc.append(self.dhw_storage._soc)
        
        heating_energy_balance = max(0, heating_energy_balance + self.sim_results['dhw_demand'][self.time_step])
        elec_demand_heating = self.dhw_heating_device.set_total_electric_consumption_heating(heat_supply = heating_energy_balance)
        self._electric_consumption_dhw_storage = elec_demand_heating - self.dhw_heating_device.get_electric_consumption_heating(heat_supply = self.sim_results['dhw_demand'][self.time_step])
        
        if self.save_memory == False:
            self.electric_consumption_dhw.append(elec_demand_heating)
            self.electric_consumption_dhw_storage.append(self._electric_consumption_dhw_storage)
        
        self.dhw_heating_device.time_step += 1
        
        return elec_demand_heating
    
        
    def set_storage_cooling(self, action):

        cooling_power_avail = self.cooling_device.get_max_cooling_power() - self.sim_results['cooling_demand'][self.time_step]
        
        if isinstance(action, list):
            action = action[0]
        cooling_energy_balance = self.cooling_storage.charge(max(-self.sim_results['cooling_demand'][self.time_step], min(cooling_power_avail, action*self.cooling_storage.capacity))) 
        
        if self.save_memory == False:
            self.cooling_device_to_storage.append(max(0, cooling_energy_balance))
            self.cooling_storage_to_building.append(-min(0, cooling_energy_balance))
            self.cooling_device_to_building.append(self.sim_results['cooling_demand'][self.time_step] + min(0, cooling_energy_balance))
            self.cooling_storage_soc.append(self.cooling_storage._soc)
        
        cooling_energy_balance = max(0, cooling_energy_balance + self.sim_results['cooling_demand'][self.time_step])
        
        elec_demand_cooling = self.cooling_device.set_total_electric_consumption_cooling(cooling_supply = cooling_energy_balance)
        
        self._electric_consumption_cooling_storage = elec_demand_cooling - self.cooling_device.get_electric_consumption_cooling(cooling_supply = self.sim_results['cooling_demand'][self.time_step])
        
        if self.save_memory == False:
            self.electric_consumption_cooling.append(np.float32(elec_demand_cooling))
            self.electric_consumption_cooling_storage.append(np.float32(self._electric_consumption_cooling_storage))
            
        self.cooling_device.time_step += 1

        return elec_demand_cooling
    

    def get_non_shiftable_load(self):
        return self.sim_results['non_shiftable_load'][self.time_step]
    
    def get_solar_power(self):
        return self.sim_results['solar_gen'][self.time_step]
    
    def get_dhw_electric_demand(self):
        return self.dhw_heating_device._electrical_consumption_heating
        
    def get_cooling_electric_demand(self):
        return self.cooling_device._electrical_consumption_cooling
    
    def reset(self):
        
        self.current_net_electricity_demand = self.sim_results['non_shiftable_load'][self.time_step] - self.sim_results['solar_gen'][self.time_step]
        
        if self.dhw_storage is not None:
            self.dhw_storage.reset()
        if self.cooling_storage is not None:
            self.cooling_storage.reset()
        if self.electrical_storage is not None:
            self.electrical_storage.reset()
        if self.dhw_heating_device is not None:
            self.dhw_heating_device.reset()
            self.current_net_electricity_demand += self.dhw_heating_device.get_electric_consumption_heating(self.sim_results['dhw_demand'][self.time_step]) 
        if self.cooling_device is not None:
            self.cooling_device.reset()
            self.current_net_electricity_demand += self.cooling_device.get_electric_consumption_cooling(self.sim_results['cooling_demand'][self.time_step])
            
        self._electric_consumption_cooling_storage = 0.0
        self._electric_consumption_dhw_storage = 0.0
        
        self.cooling_demand_building = []
        self.dhw_demand_building = []
        self.electric_consumption_appliances = []
        self.electric_generation = []
           
        self.electric_consumption_cooling = []
        self.electric_consumption_cooling_storage = []
        self.electric_consumption_dhw = []
        self.electric_consumption_dhw_storage = []
        
        self.net_electric_consumption = []
        self.net_electric_consumption_no_storage = []
        self.net_electric_consumption_no_pv_no_storage = []
        
        self.cooling_device_to_building = []
        self.cooling_storage_to_building = []
        self.cooling_device_to_storage = []
        self.cooling_storage_soc = []

        self.dhw_heating_device_to_building = []
        self.dhw_storage_to_building = []
        self.dhw_heating_device_to_storage = []
        self.dhw_storage_soc = []
        
        self.electrical_storage_electric_consumption = []
        self.electrical_storage_soc = []
        
    def terminate(self):
        
        if self.dhw_storage is not None:
            self.dhw_storage.terminate()
        if self.cooling_storage is not None:
            self.cooling_storage.terminate()
        if self.electrical_storage is not None:
            self.electrical_storage.terminate()
        if self.dhw_heating_device is not None:
            self.dhw_heating_device.terminate()
        if self.cooling_device is not None:
            self.cooling_device.terminate()
            
        if self.save_memory == False:
            
            self.cooling_demand_building = np.array(self.sim_results['cooling_demand'][:self.time_step])
            self.dhw_demand_building = np.array(self.sim_results['dhw_demand'][:self.time_step])
            self.electric_consumption_appliances = np.array(self.sim_results['non_shiftable_load'][:self.time_step])
            self.electric_generation = np.array(self.sim_results['solar_gen'][:self.time_step])
            
            elec_consumption_dhw = 0
            elec_consumption_dhw_storage = 0
            if self.dhw_heating_device.time_step == self.time_step and self.dhw_heating_device is not None:
                elec_consumption_dhw = np.array(self.electric_consumption_dhw)
                elec_consumption_dhw_storage = np.array(self.electric_consumption_dhw_storage)
                
            elec_consumption_cooling = 0
            elec_consumption_cooling_storage = 0
            if self.cooling_device.time_step == self.time_step and self.cooling_device is not None:
                elec_consumption_cooling = np.array(self.electric_consumption_cooling)
                elec_consumption_cooling_storage = np.array(self.electric_consumption_cooling_storage)
                
            self.net_electric_consumption = np.array(self.electric_consumption_appliances) + elec_consumption_cooling + elec_consumption_dhw - np.array(self.electric_generation) 
            self.net_electric_consumption_no_storage = np.array(self.electric_consumption_appliances) + (elec_consumption_cooling - elec_consumption_cooling_storage) + (elec_consumption_dhw - elec_consumption_dhw_storage) - np.array(self.electric_generation)
            self.net_electric_consumption_no_pv_no_storage = np.array(self.net_electric_consumption_no_storage) + np.array(self.electric_generation)
                
            self.cooling_demand_building = np.array(self.cooling_demand_building)
            self.dhw_demand_building = np.array(self.dhw_demand_building)
            self.electric_consumption_appliances = np.array(self.electric_consumption_appliances)
            self.electric_generation = np.array(self.electric_generation)
               
            self.electric_consumption_cooling = np.array(self.electric_consumption_cooling)
            self.electric_consumption_cooling_storage = np.array(self.electric_consumption_cooling_storage)
            self.electric_consumption_dhw = np.array(self.electric_consumption_dhw)
            self.electric_consumption_dhw_storage = np.array(self.electric_consumption_dhw_storage)
            
            self.net_electric_consumption = np.array(self.net_electric_consumption)
            self.net_electric_consumption_no_storage = np.array(self.net_electric_consumption_no_storage)
            self.net_electric_consumption_no_pv_no_storage = np.array(self.net_electric_consumption_no_pv_no_storage)
            
            self.cooling_device_to_building = np.array(self.cooling_device_to_building)
            self.cooling_storage_to_building = np.array(self.cooling_storage_to_building)
            self.cooling_device_to_storage = np.array(self.cooling_device_to_storage)
            self.cooling_storage_soc = np.array(self.cooling_storage_soc)
    
            self.dhw_heating_device_to_building = np.array(self.dhw_heating_device_to_building)
            self.dhw_storage_to_building = np.array(self.dhw_storage_to_building)
            self.dhw_heating_device_to_storage = np.array(self.dhw_heating_device_to_storage)
            self.dhw_storage_soc = np.array(self.dhw_storage_soc)
            
            self.electrical_storage_electric_consumption = np.array(self.electrical_storage_electric_consumption)
            self.electrical_storage_soc = np.array(self.electrical_storage_soc)



class WM_app:
    def __init__(self, nompow = None, eta= None, washing_time= None, duration= None, save_memory = True):

        self.nompow = nompow
        self.eta= eta
        self.max_washing= None
        self.max_WH= None
        self._cop_WH= None
        self._cop_washing= None
        self.washing_time= t_target_hotWing
        self.duration= t_target_washing
        self.t_source_WH= None
        self.t_source_washing= None
        self.cop_WH= []
        self.cop_washing= []
        self.elecal_usage_washing= []
        self.elecal_usage_WH= []
        self.hotW_supply = []
        self.washing_supply = []
        self.time_step = 0
        self.save_memory = save_memory
                   
    def get_max_washing_power(self, max_elec_power = None):

        if max_elec_power is None:
            self.max_washing= self.nompow*self.cop_washing[self.time_step]
        else:
            self.max_washing= min(max_elec_power, self.nompow)*self.cop_washing[self.time_step]
        return self.max_washing
    
    def get_max_hotWing_power(self, max_elec_power = None):

        if max_elec_power is None:
            self.max_WH= self.nompow*self.cop_washing[self.time_step]
        else:
            self.max_WH= min(max_elec_power, self.nompow)*self.cop_washing[self.time_step]
            
        return self.max_hotWing
    
    def set_total_elec_usage_washing(self, washing_supply = 0):

        
        self.washing_supply.append(washing_supply)
        self._elecal_usage_washing= washing_supply/self.cop_washing[self.time_step]
        
        if self.save_memory == False:
            self.elecal_usage_washing.append(np.float32(self._elecal_usage_washing))
            
        return self._elecal_usage_washing
            
    def get_elec_usage_washing(self, washing_supply = 0):

        _elec_usage_washing= washing_supply/self.cop_washing[self.time_step]
        return _elec_usage_washing
    
    def set_total_elec_usage_hotWing(self, hotW_supply = 0):

        self.hotW_supply.append(hotW_supply)
        self._elecal_usage_WH= hotW_supply/self.cop_hotWing[self.time_step]
        
        if self.save_memory == False:
            self.elecal_usage_hotWing.append(np.float32(self._elecal_usage_hotWing))
            
        return self._elecal_usage_hotWing
    
    def get_elec_usage_hotWing(self, hotW_supply = 0):

        _elec_usage_WH= hotW_supply/self.cop_hotWing[self.time_step]
        return _elec_usage_hotWing
    
    def reset(self):
        self.t_source_WH= None
        self.t_source_washing= None
        self.max_washing= None
        self.max_WH= None
        self._cop_WH= None
        self._cop_washing= None
        self._elecal_usage_washing= 0
        self._elecal_usage_WH= 0
        self.elecal_usage_washing= []
        self.elecal_usage_WH= []
        self.hotW_supply = []
        self.washing_supply = []
        self.time_step = 0
        
    def terminate(self):
        if self.save_memory == False:
            self.cop_WH= self.cop_hotWing[:self.time_step]
            self.cop_washing= self.cop_washing[:self.time_step]
            self.elecal_usage_washing= np.array(self.elecal_usage_washing)
            self.elecal_usage_WH= np.array(self.elecal_usage_hotWing)
            self.hotW_supply = np.array(self.hotW_supply)
            self.washing_supply = np.array(self.washing_supply) 
            
class HeatPump:
    def __init__(self, nominal_power = None, eta_tech = None, t_target_heating = None, t_target_cooling = None, save_memory = True):

        self.nominal_power = nominal_power
        self.eta_tech = eta_tech
        self.max_cooling = None
        self.max_heating = None
        self._cop_heating = None
        self._cop_cooling = None
        self.t_target_heating = t_target_heating
        self.t_target_cooling = t_target_cooling
        self.t_source_heating = None
        self.t_source_cooling = None
        self.cop_heating = []
        self.cop_cooling = []
        self.electrical_consumption_cooling = []
        self.electrical_consumption_heating = []
        self.heat_supply = []
        self.cooling_supply = []
        self.time_step = 0
        self.save_memory = save_memory
                   
    def get_max_cooling_power(self, max_electric_power = None):

        if max_electric_power is None:
            self.max_cooling = self.nominal_power*self.cop_cooling[self.time_step]
        else:
            self.max_cooling = min(max_electric_power, self.nominal_power)*self.cop_cooling[self.time_step]
        return self.max_cooling
    
    def get_max_heating_power(self, max_electric_power = None):

        if max_electric_power is None:
            self.max_heating = self.nominal_power*self.cop_cooling[self.time_step]
        else:
            self.max_heating = min(max_electric_power, self.nominal_power)*self.cop_cooling[self.time_step]
            
        return self.max_heating
    
    def set_total_electric_consumption_cooling(self, cooling_supply = 0):

        
        self.cooling_supply.append(cooling_supply)
        self._electrical_consumption_cooling = cooling_supply/self.cop_cooling[self.time_step]
        
        if self.save_memory == False:
            self.electrical_consumption_cooling.append(np.float32(self._electrical_consumption_cooling))
            
        return self._electrical_consumption_cooling
            
    def get_electric_consumption_cooling(self, cooling_supply = 0):

        _elec_consumption_cooling = cooling_supply/self.cop_cooling[self.time_step]
        return _elec_consumption_cooling
    
    def set_total_electric_consumption_heating(self, heat_supply = 0):

        self.heat_supply.append(heat_supply)
        self._electrical_consumption_heating = heat_supply/self.cop_heating[self.time_step]
        
        if self.save_memory == False:
            self.electrical_consumption_heating.append(np.float32(self._electrical_consumption_heating))
            
        return self._electrical_consumption_heating
    
    def get_electric_consumption_heating(self, heat_supply = 0):

        _elec_consumption_heating = heat_supply/self.cop_heating[self.time_step]
        return _elec_consumption_heating
    
    def reset(self):
        self.t_source_heating = None
        self.t_source_cooling = None
        self.max_cooling = None
        self.max_heating = None
        self._cop_heating = None
        self._cop_cooling = None
        self._electrical_consumption_cooling = 0
        self._electrical_consumption_heating = 0
        self.electrical_consumption_cooling = []
        self.electrical_consumption_heating = []
        self.heat_supply = []
        self.cooling_supply = []
        self.time_step = 0
        
    def terminate(self):
        if self.save_memory == False:
            self.cop_heating = self.cop_heating[:self.time_step]
            self.cop_cooling = self.cop_cooling[:self.time_step]
            self.electrical_consumption_cooling = np.array(self.electrical_consumption_cooling)
            self.electrical_consumption_heating = np.array(self.electrical_consumption_heating)
            self.heat_supply = np.array(self.heat_supply)
            self.cooling_supply = np.array(self.cooling_supply)
            
            
class PV_app:
    def __init__(self, width, height):
        self.sun= None
        self.grd= []
        self.sstrle = trle.Trle()
        self.sstrle.hidetrle()
        self.scrn= trle.Screen()
        self.ssscreen.setworldcoordinates(-width/2.0,-height/2.0,width/2.0,height/2.0)
        self.ssscreen.tracer(50)

    def addPlanet(self, a_grd):
        self.grd.append(a_grd)

    def addSun(self, asun):
        self.sun= asun


    def freeze(self):
        self.ssscreen.exitonclick()

    def moveGrd(self):
        G = .1
        dt = .001

        for p in self.grd:   
            p.moveTo(p.getXPos() + dt * p.getXVel(), p.getYPos() + dt * p.getYVel())

            rx = self.thesun.getXPos() - p.getXPos()
            ry = self.thesun.getYPos() - p.getYPos()
            r = math.sqrt(rx**2 + ry**2)

            accx = G * self.thesun.getMass()*rx/r**3
            accy = G * self.thesun.getMass()*ry/r**3

            p.setXVel(p.getXVel() + dt * accx)

            p.setYVel(p.getYVel() + dt * accy)

    def __init__(self, iname, irad, im, itemp):
            self.name = iname
            self.radius = irad
            self.mass = im
            self.temp = itemp
            self.x = 0
            self.y = 0
            self.strle = trle.Trle()


    def gnam(self):
           return self.fullname

    def gRad(self):
           return self.radius

    def gmas(self):
           return self.volume

    def temp(self):
           return self.temperture

    def getVolume(self):
           v = 5.0/2 * math.pi * self.radius**3.5


            
class FixedLoads:
    def __init__(self, nompow= None, eff= None, save_mem= True):

        self.nompow= nompow
        self.eff= efficiency
        self.max_htg= None
        self.elec_usage_htg= []
        self._elec_usage_htg= 0
        self.htg_supply = []
        self.time_step = 0
        self.save_mem= save_memory
        
    def terminate(self):
        if self.save_mem== False:
            self.elec_usage_htg= np.array(self.elec_usage_htging)
            self.htg_supply = np.array(self.htg_supply)
        
    def get_max_htgpow(self, max_electric_power = None, t_source_htg= None, t_target_htg= None):

        
        if max_electric_power is None:
            self.max_htg= self.nompow*self.efficiency
        else:
            self.max_htg= self.max_electric_power*self.efficiency
        
        return self.max_htging
    
    def set_total_electric_usage_htging(self, htg_supply = 0):

        self.htg_supply.append(htg_supply)
        self._elec_usage_htg= htg_supply/self.efficiency
        
        if self.save_mem== False:
            self.elec_usage_htging.append(np.float32(self._elec_usage_htging))
            
        return self._elec_usage_htging
    
    def get_electric_usage_htging(self, htg_supply = 0):

        _elec_usage_htg= htg_supply/self.efficiency
        return _elec_usage_htging
    
    def reset(self):
        self.max_htg= None
        self.elec_usage_htg= []
        self.htg_supply = []

        
class AC_app:
    def __init__(self, nompow= None, eff= None, save_mem= True):

        self.nompow= nompow
        self.eff= eff
        self.max_htg= None
        self.elec_usage_htg= []
        self._elec_usage_htg= 0
        self.htg_supply = []
        self.time_step = 0
        self.save_mem= save_memory
        
    def terminate(self):
        if self.save_mem== False:
            self.elec_usage_htg= np.array(self.elec_usage_htging)
            self.htg_supply = np.array(self.htg_supply)
        
    def get_max_htgpow(self, max_electric_power = None, t_source_htg= None, t_target_htg= None):

        
        if max_electric_power is None:
            self.max_htg= self.nompow*self.eff
        else:
            self.max_htg= self.max_electric_power*self.eff
        
        return self.max_htging
    
    def set_total_electric_usage_htging(self, htg_supply = 0):

        self.htg_supply.append(htg_supply)
        self._elec_usage_htg= htg_supply/self.eff
        
        if self.save_mem== False:
            self.elec_usage_htging.append(np.float32(self._elec_usage_htging))
            
        return self._elec_usage_htging
    
    def get_electric_usage_htging(self, htg_supply = 0):

        _elec_usage_htg= htg_supply/self.eff
        return _elec_usage_htging
    
    def reset(self):
        self.max_htg= None
        self.elec_usage_htg= []
        self.htg_supply = []
     

    
class ElectricHeater:
    def __init__(self, nominal_power = None, efficiency = None, save_memory = True):

        self.nominal_power = nominal_power
        self.efficiency = efficiency
        self.max_heating = None
        self.electrical_consumption_heating = []
        self._electrical_consumption_heating = 0
        self.heat_supply = []
        self.time_step = 0
        self.save_memory = save_memory
        
    def terminate(self):
        if self.save_memory == False:
            self.electrical_consumption_heating = np.array(self.electrical_consumption_heating)
            self.heat_supply = np.array(self.heat_supply)
        
    def get_max_heating_power(self, max_electric_power = None, t_source_heating = None, t_target_heating = None):

        
        if max_electric_power is None:
            self.max_heating = self.nominal_power*self.efficiency
        else:
            self.max_heating = self.max_electric_power*self.efficiency
        
        return self.max_heating
    
    def set_total_electric_consumption_heating(self, heat_supply = 0):

        self.heat_supply.append(heat_supply)
        self._electrical_consumption_heating = heat_supply/self.efficiency
        
        if self.save_memory == False:
            self.electrical_consumption_heating.append(np.float32(self._electrical_consumption_heating))
            
        return self._electrical_consumption_heating
    
    def get_electric_consumption_heating(self, heat_supply = 0):

        _electrical_consumption_heating = heat_supply/self.efficiency
        return _electrical_consumption_heating
    
    def reset(self):
        self.max_heating = None
        self.electrical_consumption_heating = []
        self.heat_supply = []
    
    
class Dryer_app:
    def __init__(self, nompow = None, eta= None, dry_time= None, duration= None, save_memo= True):

        self.nompow = nompow
        self.eta= eta
        self.max_dry= None
        self.max_DRY= None
        self._cop_DRY= None
        self._cop_dry= None
        self.dry_time= t_target_hotWing
        self.duration= t_target_dry
        self.t_source_DRY= None
        self.t_source_dry= None
        self.cop_DRY= []
        self.cop_dry= []
        self.elecal_usage_dry= []
        self.elecal_usage_DRY= []
        self.hotW_supply = []
        self.dry_supply = []
        self.time_step = 0
        self.save_memo= save_memory
                   
    def get_max_dry_power(self, max_elec_power = None):

        if max_elec_power is None:
            self.max_dry= self.nompow*self.cop_dry[self.time_step]
        else:
            self.max_dry= min(max_elec_power, self.nompow)*self.cop_dry[self.time_step]
        return self.max_dry
    
    def get_max_hotWing_power(self, max_elec_power = None):

        if max_elec_power is None:
            self.max_DRY= self.nompow*self.cop_dry[self.time_step]
        else:
            self.max_DRY= min(max_elec_power, self.nompow)*self.cop_dry[self.time_step]
            
        return self.max_hotWing
    
    def set_total_elec_usage_dry(self, dry_supply = 0):

        
        self.dry_supply.append(dry_supply)
        self._elecal_usage_dry= dry_supply/self.cop_dry[self.time_step]
        
        if self.save_memo== False:
            self.elecal_usage_dry.append(np.float32(self._elecal_usage_dry))
            
        return self._elecal_usage_dry
            
    def get_elec_usage_dry(self, dry_supply = 0):

        _elec_usage_dry= dry_supply/self.cop_dry[self.time_step]
        return _elec_usage_dry
    
    def set_total_elec_usage_hotWing(self, hotW_supply = 0):

        self.hotW_supply.append(hotW_supply)
        self._elecal_usage_DRY= hotW_supply/self.cop_hotWing[self.time_step]
        
        if self.save_memo== False:
            self.elecal_usage_hotWing.append(np.float32(self._elecal_usage_hotWing))
            
        return self._elecal_usage_hotWing
    
    def get_elec_usage_hotWing(self, hotW_supply = 0):

        _elec_usage_DRY= hotW_supply/self.cop_hotWing[self.time_step]
        return _elec_usage_hotWing
    
    def reset(self):
        self.t_source_DRY= None
        self.t_source_dry= None
        self.max_dry= None
        self.max_DRY= None
        self._cop_DRY= None
        self._cop_dry= None
        self._elecal_usage_dry= 0
        self._elecal_usage_DRY= 0
        self.elecal_usage_dry= []
        self.elecal_usage_DRY= []
        self.hotW_supply = []
        self.dry_supply = []
        self.time_step = 0
        
    def terminate(self):
        if self.save_memo== False:
            self.cop_DRY= self.cop_hotWing[:self.time_step]
            self.cop_dry= self.cop_dry[:self.time_step]
            self.elecal_usage_dry= np.array(self.elecal_usage_dry)
            self.elecal_usage_DRY= np.array(self.elecal_usage_hotWing)
            self.hotW_supply = np.array(self.hotW_supply)
            self.dry_supply = np.array(self.dry_supply)

            
class EnergyStorage:
    def __init__(self, capacity = None, max_power_output = None, max_power_charging = None, efficiency = 1, loss_coef = 0, save_memory = True):

        self.capacity = capacity
        self.max_power_output = max_power_output
        self.max_power_charging = max_power_charging
        self.efficiency = efficiency**0.5
        self.loss_coef = loss_coef
        self.soc = []
        self._soc = 0 
        self.energy_balance = []
        self._energy_balance = 0
        self.save_memory = save_memory
        
    def terminate(self):
        if self.save_memory == False:
            self.energy_balance = np.array(self.energy_balance)
            self.soc =  np.array(self.soc)
        
    def charge(self, energy):

        soc_init = self._soc*(1-self.loss_coef)
        
    
        if energy >= 0:
            if self.max_power_charging is not None:
                energy =  min(energy, self.max_power_charging)
            self._soc = soc_init + energy*self.efficiency
            
        else:
            if self.max_power_output is not None:
                energy = max(-max_power_output, energy)
            self._soc = max(0, soc_init + energy/self.efficiency)  
            
        if self.capacity is not None:
            self._soc = min(self._soc, self.capacity)
          
        
        if energy >= 0:
            self._energy_balance = (self._soc - soc_init)/self.efficiency
            
        else:
            self._energy_balance = (self._soc - soc_init)*self.efficiency
        
        if self.save_memory == False:
            self.energy_balance.append(np.float32(self._energy_balance))
            self.soc.append(np.float32(self._soc))
            
        return self._energy_balance
    
    def reset(self):
        self.soc = []
        self._soc = 0 
        self.energy_balance = [] 
        self._energy_balance = 0
        self.time_step = 0

        
class Battery:
    def __init__(self, capacity, nominal_power = None, capacity_loss_coef = None, power_efficiency_curve = None, capacity_power_curve = None, efficiency = None, loss_coef = 0, save_memory = True):

        self.capacity = capacity
        self.c0 = capacity
        self.nominal_power = nominal_power
        self.capacity_loss_coef = capacity_loss_coef
        
        if power_efficiency_curve is not None:
            self.power_efficiency_curve = np.array(power_efficiency_curve).T
        else:
            self.power_efficiency_curve = power_efficiency_curve
            
        if capacity_power_curve is not None:
            self.capacity_power_curve = np.array(capacity_power_curve).T
        else:
            self.capacity_power_curve = capacity_power_curve
            
        self.efficiency = efficiency**0.5
        self.loss_coef = loss_coef
        self.max_power = None
        self._eff = []
        self._energy = []
        self._max_power = []
        self.soc = []
        self._soc = 0 
        self.energy_balance = []
        self._energy_balance = 0
        self.save_memory = save_memory
        
    def terminate(self):
        if self.save_memory == False:
            self.energy_balance = np.array(self.energy_balance)
            self.soc =  np.array(self.soc)
        
    def charge(self, energy):

        soc_init = self._soc*(1-self.loss_coef)
        if self.capacity_power_curve is not None:
            soc_normalized = soc_init/self.capacity
            idx = max(0, np.argmax(soc_normalized <= self.capacity_power_curve[0]) - 1)
            
            self.max_power = self.nominal_power*(self.capacity_power_curve[1][idx] + (self.capacity_power_curve[1][idx+1] - self.capacity_power_curve[1][idx]) * (soc_normalized - self.capacity_power_curve[0][idx])/(self.capacity_power_curve[0][idx+1] - self.capacity_power_curve[0][idx]))
        
        else:
            self.max_power = self.nominal_power
          
        if energy >= 0:
            if self.nominal_power is not None:
                
                energy =  min(energy, self.max_power)
                if self.power_efficiency_curve is not None:
                    # Calculating the maximum power rate at which the battery can be charged or discharged
                    energy_normalized = np.abs(energy)/self.nominal_power
                    idx = max(0, np.argmax(energy_normalized <= self.power_efficiency_curve[0]) - 1)
                    self.efficiency = self.power_efficiency_curve[1][idx] + (energy_normalized - self.power_efficiency_curve[0][idx])*(self.power_efficiency_curve[1][idx + 1] - self.power_efficiency_curve[1][idx])/(self.power_efficiency_curve[0][idx + 1] - self.power_efficiency_curve[0][idx])
                    self.efficiency = self.efficiency**0.5
                 
            self._soc = soc_init + energy*self.efficiency
            
        else:
            if self.nominal_power is not None:
                energy = max(-self.max_power, energy)
                
            if self.power_efficiency_curve is not None:
                
                energy_normalized = np.abs(energy)/self.nominal_power
                idx = max(0, np.argmax(energy_normalized <= self.power_efficiency_curve[0]) - 1)
                self.efficiency = self.power_efficiency_curve[1][idx] + (energy_normalized - self.power_efficiency_curve[0][idx])*(self.power_efficiency_curve[1][idx + 1] - self.power_efficiency_curve[1][idx])/(self.power_efficiency_curve[0][idx + 1] - self.power_efficiency_curve[0][idx])
                self.efficiency = self.efficiency**0.5
                    
            self._soc = max(0, soc_init + energy/self.efficiency)
            
        if self.capacity is not None:
            self._soc = min(self._soc, self.capacity)
          
        if energy >= 0:
            self._energy_balance = (self._soc - soc_init)/self.efficiency
            
        else:
            self._energy_balance = (self._soc - soc_init)*self.efficiency
            
        self.capacity -= self.capacity_loss_coef*self.c0*np.abs(self._energy_balance)/(2*self.capacity)
        
        if self.save_memory == False:
            self.energy_balance.append(np.float32(self._energy_balance))
            self.soc.append(np.float32(self._soc))
            
        return self._energy_balance
    
    def reset(self):
        self.soc = []
        self._soc = 0 
        self.energy_balance = [] 
        self._energy_balance = 0
        self.time_step = 0
        
class DW_app:
    def __init__(self, nompow = None, eta= None, washing_time= None, duration= None, save_memory = True):

        self.nompow = nompow
        self.eta= eta
        self.max_washing= None
        self.max_WH= None
        self._cop_WH= None
        self._cop_washing= None
        self.washing_time= t_target_hotWing
        self.duration= t_target_washing
        self.t_source_WH= None
        self.t_source_washing= None
        self.cop_WH= []
        self.cop_washing= []
        self.elecal_usage_washing= []
        self.elecal_usage_WH= []
        self.hotW_supply = []
        self.washing_supply = []
        self.time_step = 0
        self.save_memory = save_memory
                   
    def get_max_washing_power(self, max_elec_power = None):

        if max_elec_power is None:
            self.max_washing= self.nompow*self.cop_washing[self.time_step]
        else:
            self.max_washing= min(max_elec_power, self.nompow)*self.cop_washing[self.time_step]
        return self.max_washing
    
    def get_max_hotWing_power(self, max_elec_power = None):

        if max_elec_power is None:
            self.max_WH= self.nompow*self.cop_washing[self.time_step]
        else:
            self.max_WH= min(max_elec_power, self.nompow)*self.cop_washing[self.time_step]
            
        return self.max_hotWing
    
    def set_total_elec_usage_washing(self, washing_supply = 0):

        
        self.washing_supply.append(washing_supply)
        self._elecal_usage_washing= washing_supply/self.cop_washing[self.time_step]
        
        if self.save_memory == False:
            self.elecal_usage_washing.append(np.float32(self._elecal_usage_washing))
            
        return self._elecal_usage_washing
            
    def get_elec_usage_washing(self, washing_supply = 0):

        _elec_usage_washing= washing_supply/self.cop_washing[self.time_step]
        return _elec_usage_washing
    
    def set_total_elec_usage_hotWing(self, hotW_supply = 0):

        self.hotW_supply.append(hotW_supply)
        self._elecal_usage_WH= hotW_supply/self.cop_hotWing[self.time_step]
        
        if self.save_memory == False:
            self.elecal_usage_hotWing.append(np.float32(self._elecal_usage_hotWing))
            
        return self._elecal_usage_hotWing
    
    def get_elec_usage_hotWing(self, hotW_supply = 0):

        _elec_usage_WH= hotW_supply/self.cop_hotWing[self.time_step]
        return _elec_usage_hotWing
    
    def reset(self):
        self.t_source_WH= None
        self.t_source_washing= None
        self.max_washing= None
        self.max_WH= None
        self._cop_WH= None
        self._cop_washing= None
        self._elecal_usage_washing= 0
        self._elecal_usage_WH= 0
        self.elecal_usage_washing= []
        self.elecal_usage_WH= []
        self.hotW_supply = []
        self.washing_supply = []
        self.time_step = 0
        
    def terminate(self):
        if self.save_memory == False:
            self.cop_WH= self.cop_hotWing[:self.time_step]
            self.cop_washing= self.cop_washing[:self.time_step]
            self.elecal_usage_washing= np.array(self.elecal_usage_washing)
            self.elecal_usage_WH= np.array(self.elecal_usage_hotWing)
            self.hotW_supply = np.array(self.hotW_supply)
            self.washing_supply = np.array(self.washing_supply) 


# In[4]:


class Environment(object):
    def __init__(self, optimal_temperature = [18.0, 24.0], initial_month= 0,                  initial_number_users = 10, initial_rate_data = 60):
        
        self.initial_month = initial_month
        self.monthly_atmospheric_temperatures = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0, 
                                                 23.0, 24.0, 22.0, 10.0, 5.0, 1.0]
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[initial_month]
        self.optimal_temperature = optimal_temperature
        self.min_temperature = -20
        self.max_temperature = 80
        self.min_number_users = 10
        self.max_number_users = 100
        self.max_update_users = 5
        self.initial_number_users = initial_number_users
        self.current_number_users = initial_number_users
        self.min_rate_data = 20
        self.max_rate_data = 300
        self.max_update_data = 10
        self.initial_rate_data = initial_rate_data
        self.current_rate_data = initial_rate_data
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users                                     + 1.25 * self.current_rate_data
        self.temperature_Routing = self.intrinsic_temperature
        self.temperature_noRouting = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0 # mid of optimal range
        self.total_energy_Routing = 0.0
        self.total_energy_noRouting = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1        
        
    def update_env(self, direction, energy_Routing, month):

        
        energy_noRouting = 0
        if (self.temperature_noRouting < self.optimal_temperature[0]):
            energy_noRouting = self.optimal_temperature[0] - self.temperature_noRouting
            self.temperature_noRouting = self.optimal_temperature[0]
        elif (self.temperature_noRouting > self.optimal_temperature[1]):
            energy_noRouting = self.temperature_noRouting - self.optimal_temperature[1]
            self.temperature_noRouting = self.optimal_temperature[1]
        self.reward = energy_noRouting - energy_Routing
        self.reward = 1e-3 * self.reward
        
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[month]
        self.current_number_users += np.random.randint(-self.max_update_users, self.max_update_users)
        if (self.current_number_users > self.max_number_users):
            self.current_number_users = self.max_number_users
        elif (self.current_number_users < self.min_number_users):
            self.current_number_users = self.min_number_users
        self.current_rate_data += np.random.randint(-self.max_update_data, self.max_update_data)
        if (self.current_rate_data > self.max_rate_data):
            self.current_rate_data = self.max_rate_data
        elif (self.current_rate_data < self.min_rate_data):
            self.current_rate_data = self.min_rate_data
        past_intrinsic_temperature = self.intrinsic_temperature      
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users                                      + 1.25 * self.current_rate_data  
        delta_intrinsic_temperature = self.intrinsic_temperature - past_intrinsic_temperature
        if (direction == -1):
            delta_temperature_Routing = -energy_Routing  
        elif (direction == 1):
            delta_temperature_Routing = energy_Routing
        self.temperature_Routing += delta_intrinsic_temperature + delta_temperature_Routing
        self.temperature_noRouting += delta_intrinsic_temperature
        
        if (self.temperature_Routing < self.min_temperature):
            if self.train == 1:
                self.game_over = 1
            else:
                self.total_energy_Routing += self.optimal_temperature[0] - self.temperature_Routing
                self.temperature_Routing = self.optimal_temperature[0]
        elif (self.temperature_Routing > self.max_temperature):   
            if self.train == 1:
                self.game_over = 1
            else:
                self.total_energy_Routing += self.temperature_Routing - self.optimal_temperature[1] 
                self.temperature_Routing = self.optimal_temperature[1]
        
        self.total_energy_Routing += energy_Routing
        self.total_energy_noRouting += energy_noRouting
        
        scaled_temperature_Routing = (self.temperature_Routing - self.min_temperature) /                                 (self.max_temperature - self.min_temperature)
        scaled_number_users = (self.current_number_users - self.min_number_users) /                               (self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data) /                            (self.max_rate_data - self.min_rate_data)
        next_state = np.matrix([scaled_temperature_Routing, scaled_number_users, scaled_rate_data])
        
        return next_state, self.reward, self.game_over 
    
    def reset(self, new_month):
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[new_month]
        self.initial_month = new_month
        self.current_number_users = self.initial_number_users
        self.current_rate_data = self.initial_rate_data
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users                                      + 1.25 * self.current_rate_data
        self.temperature_Routing = self.intrinsic_temperature
        self.temperature_noRouting = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
        self.total_energy_Routing = 0.0
        self.total_energy_noRouting = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1

    def observe(self):
        scaled_temperature_Routing = (self.temperature_Routing - self.min_temperature) /                                 (self.max_temperature - self.min_temperature)
        scaled_number_users = (self.current_number_users - self.min_number_users) /                               (self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data) /                            (self.max_rate_data - self.min_rate_data)
        current_state = np.matrix([scaled_temperature_Routing, scaled_number_users, scaled_rate_data])
        
        return current_state, self.reward, self.game_over
    
    
    

class Env:
    def __init__(self, Device_id, Device_usage, Energy_charge, udc, from_timeslotnumber, to_timeslotnumber
                 , consumption_period, penalty, incentive):
        self.Device_id = Device_id
        self.Device_usage = Device_usage
        self.Energy_charge = Energy_charge
        self.udc = udc
        self.from_timeslotnumber = from_timeslotnumber
        self.to_timeslotnumber = to_timeslotnumber
        self.consumption_period = consumption_period
        self.penalty = penalty
        self.incentive = incentive
        self.preferences_satisfied = True
        self.done = False
        self.time_stamp = 0
        self.state_accumulation = 0
        self.episode_rewards = []
        self.history_actions = []
#         print(f'schedule: {self.from_timeslotnumber} - {self.to_timeslotnumber}, duration: {self.consumption_period}')

    def get_action_shape(self):
        return 1

    def reset(self):
        self.done = False
        self.time_stamp = 0
        self.state_accumulation = 0
        self.history_actions = []
        return self.get_obs()

    def action_space_sample(self):
        return [random.randint(0, 1)]

    def get_obs_shape(self):
        return np.shape(self.get_obs())

    def get_obs(self):
        return [self.time_stamp, self.state_accumulation, self.from_timeslotnumber,
                self.consumption_period, self.to_timeslotnumber]

    def reward(self, action):
        under_schedule= self.from_timeslotnumber <= self.time_stamp < self.to_timeslotnumber
        at_to_timeslotnumber = self.time_stamp == self.to_timeslotnumber

        reward_function = (1 - under_schedule) *                           (at_to_timeslotnumber * self.incentive * (self.consumption_period -                                                                 np.abs(self.consumption_period - self.state_accumulation)) +                            action * self.penalty +
                           (1 - action) * self.incentive) + \
                          under_schedule* (
                                  action * (self.Energy_charge[self.time_stamp] * self.Device_usage) + \
                                  (1 - action) * self.udc * self.Device_usage)
        reward_function *= -1
        self.episode_rewards.append(reward_function)
        return reward_function

    def old_reward(self, action):
        under_schedule = self.from_timeslotnumber <= self.time_stamp <= self.to_timeslotnumber
        at_to_timeslotnumber = self.time_stamp == self.to_timeslotnumber

        reward_function = (1 - under_schedule) *                           (action * self.penalty +
                           (1 - action) * self.incentive) + \
                          under_schedule* (
                                  at_to_timeslotnumber * ((not self.preferences_satisfied) *
                                                      self.penalty *
                                                      np.abs(self.consumption_period - self.state_accumulation) +
                                                      self.preferences_satisfied *
                                                      self.incentive * self.consumption_period) + \
                                  (1 - at_to_timeslotnumber) *
                                  (action * self.Energy_charge[self.time_stamp] * self.Device_usage + \
                                   (1 - action) * self.udc * self.Device_usage))
        reward_function *= -1
        self.episode_rewards.append(reward_function)
        return reward_function

    def step(self, action):


        self.history_actions.append(action)
        self.state_accumulation += action
        if self.state_accumulation != self.consumption_period and self.time_stamp == self.to_timeslotnumber:
            self.preferences_satisfied = False
        reward = self.reward(action)
        self.time_stamp += 1
        self.done = self.time_stamp == 24
        return self.get_obs(), reward, self.done, None


def get_random_env():
    Device_id = 1
    udc = 0.
    Energy_charge = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 12, 12, 5, 5, 5, 5, 10, 10, 10, 5, 5, 5])
    Device_usage = 1
    from_timeslotnumber = 5
    to_timeslotnumber = 20
    consumption_period = 4
    penalty = 10.
    incentive = -10.

    return Env(Device_id, Device_usage, Energy_charge, udc, from_timeslotnumber, to_timeslotnumber,
                       consumption_period, penalty, incentive)


# In[5]:


class Buffer:
    def __init__(self):
        self.buffer = []
        
    def append_sample(self, sample):
        self.buffer.append(sample)
        
    def sample(self, sample_size):
        s, a, r, s_next, done = [],[],[],[],[]
        
        if sample_size > len(self.buffer):
            sample_size = len(self.buffer)
            
        rand_sample = random.sample(self.buffer, sample_size)
        for values in rand_sample:
            s.append(values[0])
            a.append(values[1])
            r.append(values[2])
            s_next.append(values[3])
            done.append([4])
        return torch.tensor(s,dtype=torch.float32), torch.tensor(a,dtype=torch.float32), torch.tensor(r,dtype=torch.float32), torch.tensor(s_next,dtype=torch.float32), done
    
    def __len__(self):
         return len(self.buffer)
                           
class Actor():   
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 5)
        self.l2 = nn.Linear(5, 3)
        self.l3 = nn.Linear(3, action_dim)

        self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic():
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 7)
        self.l2 = nn.Linear(7, 6)
        self.l3 = nn.Linear(6, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 7)
        self.l5 = nn.Linear(7, 6)
        self.l6 = nn.Linear(6, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    
class RL_Agents:
    def __init__(self, building_info, observation_spaces = None, action_spaces = None):
        
        self.discount = 0.992 
        self.batch_size = 100 
        self.iterations = 1 
        self.policy_freq = 2 
        self.tau = 5e-3 
        self.lr_init = 1e-3 
        self.lr_final = 1e-3 
        self.lr_decay_rate = 1/(78*8760)
        self.expl_noise_init = 0.75 
        self.expl_noise_final = 0.01 
        self.expl_noise_decay_rate = 1/(290*8760) 
        self.policy_noise = 0.025*0
        self.noise_clip = 0.04*0
        self.max_action = 0.25
        self.min_samples_training = 400 
        self.device = "cpu"
        self.time_step = 0
        self.building_info = building_info 
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.n_buildings = len(observation_spaces)
        self.buffer = {i: Buffer() for i in range(self.n_buildings)}
        self.networks_initialized = False
        
        self.actor_loss_list = {i: [] for i in range(self.n_buildings)}
        self.critic1_loss_list = {i: [] for i in range(self.n_buildings)}
        self.critic2_loss_list = {i: [] for i in range(self.n_buildings)}
        self.q_val_list = {i: [] for i in range(self.n_buildings)}
        self.q1_list = {i: [] for i in range(self.n_buildings)}
        self.q2_list = {i: [] for i in range(self.n_buildings)}
        self.a_track1 = []
        self.a_track2 = []
        
        self.actor, self.critic, self.actor_target, self.critic_target, self.actor_optimizer, self.critic_optimizer =  {}, {}, {}, {}, {}, {}
        for i, (o, a) in enumerate(zip(observation_spaces, action_spaces)):
            self.actor[i] = Actor(o.shape[0], a.shape[0], self.max_action).to(self.device)
            self.critic[i] = Critic(o.shape[0], a.shape[0]).to(self.device)
            self.actor_target[i] = copy.deepcopy(self.actor[i])
            self.critic_target[i] = copy.deepcopy(self.critic[i])
            self.actor_optimizer[i] = optim.Adam(self.actor[i].parameters(), lr=self.lr_init)
            self.critic_optimizer[i] = optim.Adam(self.critic[i].parameters(), lr=self.lr_init)
        
    def select_action(self, states):
        expl_noise = max(self.expl_noise_final, self.expl_noise_init * (1 - self.time_step * self.expl_noise_decay_rate))
        
        actions = []
        for i, state in enumerate(states):
            a = self.actor[i](torch.tensor(state, dtype=torch.float32))
            self.a_track1.append(a)
            a = a.cpu().detach().numpy() + expl_noise * np.random.normal(loc = 0, scale = self.max_action, size=a.shape)
            self.a_track2.append(a)
            a = np.clip(a, -self.max_action, self.max_action)
            actions.append(a)
        return actions
    
    def add_to_buffer(self, states, actions, rewards, next_states, dones):
        
        dones = [dones for _ in range(self.n_buildings)]
        
        for i, (s, a, r, s_next, done) in enumerate(zip(states, actions, rewards, next_states, dones)):
            s = (s - self.observation_spaces[i].low)/(self.observation_spaces[i].high - self.observation_spaces[i].low + 0.00001)
            s_next = (s_next - self.observation_spaces[i].low)/(self.observation_spaces[i].high - self.observation_spaces[i].low + 0.00001)
            self.buffer[i].append_sample((s, a, r, s_next, done))

        lr = max(self.lr_final, self.lr_init * (1 - self.time_step * self.lr_decay_rate))
        for i in range(self.n_buildings):
            self.actor_optimizer[i] = optim.Adam(self.actor[i].parameters(), lr=lr)
            self.critic_optimizer[i] = optim.Adam(self.critic[i].parameters(), lr=lr)
            
        for i in range(self.n_buildings):
            
            if len(self.buffer[i]) > self.min_samples_training:
                
                for k in range(self.iterations):
                    state, action, reward, next_state, dones_mask = self.buffer[i].sample(self.batch_size)
                    target_Q = reward.unsqueeze(dim=-1)

                    with torch.no_grad():
                        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                        
                        next_action = (self.actor_target[i](next_state) + noise).clamp(-self.max_action, self.max_action)
                        
                        target_Q1, target_Q2 = self.critic_target[i](next_state, next_action)
                        target_Q = torch.min(target_Q1, target_Q2)
                        target_Q = reward.unsqueeze(dim=-1) + target_Q * self.discount
                        
                    current_Q1, current_Q2 = self.critic[i](state, action)    
                    
                    critic1_loss = F.mse_loss(current_Q1, target_Q)
                    critic2_loss = F.mse_loss(current_Q2, target_Q)
                    critic_loss = critic1_loss + critic2_loss
                    
                    self.critic_optimizer[i].zero_grad()
                    critic_loss.backward()  
                    self.critic_optimizer[i].step()
                    
                    self.q_val_list[i].append(target_Q)
                    self.q1_list[i].append(current_Q1)
                    self.q2_list[i].append(current_Q2)
                    self.critic1_loss_list[i].append(critic1_loss)
                    self.critic2_loss_list[i].append(critic2_loss)
                    
                    if k % self.policy_freq == 0:
                        
                        actor_loss = -self.critic[i].Q1(state, self.actor[i](state)).mean()
                        self.actor_loss_list[i].append(actor_loss)
                                        
                        self.actor_optimizer[i].zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer[i].step()

                        for param, target_param in zip(self.critic[i].parameters(), self.critic_target[i].parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                        for param, target_param in zip(self.actor[i].parameters(), self.actor_target[i].parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        self.time_step += 1
                            
            
class RBC_Agent:
    def __init__(self, actions_spaces):
        self.actions_spaces = actions_spaces
        self.reset_action_tracker()
        
    def reset_action_tracker(self):
        self.action_tracker = []
        
    def select_action(self, states):
        hour_day = states[0][0]
        
        a = [[0.0 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        if hour_day >= 9 and hour_day <= 21:
            a = [[-0.08 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        
        if (hour_day >= 1 and hour_day <= 8) or (hour_day >= 22 and hour_day <= 24):
            a = []
            for i in range(len(self.actions_spaces)):
                if len(self.actions_spaces[i].sample()) == 2:
                    a.append([0.091, 0.091])
                else:
                    a.append([0.091])

        self.action_tracker.append(a)
        return np.array(a)
    

class DQN(object):
    
    def __init__(self, max_memory = 100, discount = 0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount = discount    

    def remember(self, transition, game_over):

        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]                   

    def get_batch(self, model, batch_size = 10):
        len_memory = len(self.memory)
        num_inputs = self.memory[0][0][0].shape[1]  
        num_outputs = model.output_shape[-1]
        
        inputs = np.zeros((min(len_memory, batch_size), num_inputs))   
        targets = np.zeros((min(len_memory, batch_size), num_outputs)) 
        
        for i, idx in enumerate(np.random.randint(0, len_memory, size = min(len_memory, batch_size))):
            current_state, action, reward, next_state = self.memory[idx][0]
            game_over = self.memory[idx][1]
            inputs[i] = current_state
            targets[i] = model.predict(current_state)[0]  
            Q_sa = np.max(targets[i])
            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * Q_sa
        
        return inputs, targets
    
    
class DeepNN(object):
    def __init__(self, learning_rate = 0.001, number_actions = 5):
        self.learning_rate = learning_rate
        self.number_actions = number_actions
        states = Input(shape = (3,))
        x = Dense(units = 64, activation = 'sigmoid')(states)
        x = Dense(units = 32, activation = 'sigmoid')(x)
        q_values = Dense(units = self.number_actions, activation = 'softmax')(x)
        
        self.model = Model(inputs = states, outputs = q_values)
        self.model.compile(loss='mse', optimizer = Adam(lr=self.learning_rate))


# In[7]:


os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)

epsilon = .3   
number_actions = 4
direction_boundary = (number_actions - 1) / 3   
number_epochs = 100
max_memory = 300
batch_size = 2048
temperature_step = 1.5

env = Environment(optimal_temperature = (18.0, 24.0), initial_month = 0,                   initial_number_users = 20, initial_rate_data = 30)

dnn = DeepNN(learning_rate = 0.00001, number_actions = number_actions)

dqn = DQN(max_memory = max_memory, discount = 0.9)

train = True

env.train = train
model = dnn.model
early_stopping = True
patience = 10
best_total_reward = -np.inf
patience_count = 0

if (env.train):
    
    for epoch in range(1, number_epochs):
        
        total_reward = 0
        loss = 0.
        new_month = np.random.randint(0, 12)
        env.reset(new_month = new_month)
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0
        
        while ((not game_over) and timestep <= 5 * 30 * 24 * 60):
            
            if np.random.rand() <= epsilon:   
                action = np.random.randint(0, number_actions)  
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                    direction = 1
                energy_Routing = abs(action - direction_boundary) * temperature_step
            
            else:
                q_values = model.predict(current_state)
                action = np.argmax(q_values[0])
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                    direction = 1
                energy_Routing = abs(action - direction_boundary) * temperature_step
            
            next_state, reward, game_over = env.update_env(direction, energy_Routing,                                                            int(timestep / (30 * 24 * 60))) 
            total_reward += reward
            
            dqn.remember([current_state, action, reward, next_state], game_over)
            
            inputs, targets = dqn.get_batch(model, batch_size = batch_size)
            
            loss += model.train_on_batch(inputs, targets)  
            timestep += 1
            current_state = next_state                     
        
        print("Epoch: {:03d}/{:03d}".format(epoch, number_epochs))
        print("Regular Consumption: {:.0f}".format(env.total_energy_noRouting))
        print("Consumption using smart routing: {:.0f}".format(env.total_energy_Routing))
        
        
        model.save("model.h5")


print('Evaluating one year of energy management...')
env = Environment(optimal_temperature = (18.0, 24.0), initial_month = 0,                   initial_number_users = 20, initial_rate_data = 30)
model = load_model("model.h5")
train = False
env.train = train
current_state, _, _ = env.observe()

for timestep in tqdm(range(1 * 30 * 24 * 60)):
    q_values = model.predict(current_state)
    action = np.argmax(q_values[0])
    if (action - direction_boundary < 0):
        direction = -1
    else:
        direction = 1
    energy_Routing = abs(action - direction_boundary) * temperature_step
    next_state, _, _ = env.update_env(direction, energy_Routing,                                                 int(timestep / (30 * 24 * 60)))  # month [0,11]         
    current_state = next_state   

print("Regular Consumption: {:.0f}".format(env.total_energy_noRouting))
print("Consumption using smart routing: {:.0f}".format(env.total_energy_Routing))
print("ENERGY SAVED WITH AI: {:.0f}%".format((env.total_energy_noRouting - env.total_energy_Routing)/env.total_energy_noRouting*100))