# --- CLASS NAME MAPPING ---
# This dictionary maps the raw model output to the formal name for the LLM.
# KEY: The exact class name from your model's data.yaml
# VALUE: The clean, formal name you want the LLM to use

CLASS_NAME_MAP = {
    '4by3 hand lever valve': '4/3 Way Pneumatic Hand Lever Valve',
    '5by2 DCV': '5/2 Way Directional Control Valve',
    'FRL unit': 'Pneumatic FRL Unit (Filter, Regulator, Lubricator)',
    'agriculturedrone': 'Agricultural Drone',
    'double acting pneumatic cylinder': 'Double-Acting Pneumatic Cylinder',
    'inductive sensor': 'Inductive Proximity Sensor',
    'load': 'Mechanical Load', # You can make this more specific if needed
    'push button': 'Industrial Push Button Switch',
    'quadcopter': 'Quadcopter Drone'
}

def get_formal_name(informal_name: str) -> str:
    """
    Translates a raw model class name into its formal, human-readable name.
    
    If the name isn't in our map, it will just clean it up and return it
    as a fallback.
    """
    if not informal_name:
        return "Unknown Equipment"

    # 1. Try to find the formal name in our map
    #    .get() is a safe way to check a dictionary
    formal_name = CLASS_NAME_MAP.get(informal_name)
    
    if formal_name:
        return formal_name
    
    # 2. If no formal name is found, use the old method as a fallback
    #    (e.g., if you add a new class but forget to update the map)
    return informal_name.replace('_', ' ').title()