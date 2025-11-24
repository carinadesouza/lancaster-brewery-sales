import pandas as pd
import folium
from folium import plugins
import pgeocode


df = pd.read_excel("LB SALES.xlsx")
df['OutletPostCode'] = df['OutletPostCode'].astype(str).str.strip().str.upper()

key_cols = ['Price Paid', 'Total Cost Price', 'Product Cost Price', 'Discount', 'Duty Element', 'Qty']

# Get lat/lon for postcodes
geo = pgeocode.Nominatim('GB') # Set up UK postcode lookup.
def get_coords(pc):
    result = geo.query_postal_code(pc) # looks for the post code 
    if result is None or pd.isna(result.latitude) or pd.isna(result.longitude):
        return pd.Series({'lat': None, 'lon': None})
    return pd.Series({'lat': result.latitude, 'lon': result.longitude}) # return lat and long 

coords = df['OutletPostCode'].apply(get_coords)
df = pd.concat([df, coords], axis=1)

# Keep only valid coordinates
df_geo = df.dropna(subset=['lat', 'lon'])

# Aggregate data by postcode
agg_dict = {col: 'sum' for col in key_cols}
agg_dict['lat'] = 'first'
agg_dict['lon'] = 'first'

df_agg = df_geo.groupby('OutletPostCode').agg(agg_dict).reset_index()

# Create function to generate maps for each metric
def create_sales_map(df_agg, metric_col, filename):
    # Create base map centered on Lancaster
    m = folium.Map(
        location=[54.0470, -2.8010],  # Lancaster coordinates
        zoom_start=8,
        tiles='OpenStreetMap'
    )
    
    # Normalize the metric for sizing
    max_val = df_agg[metric_col].max()
    min_val = df_agg[metric_col].min()
    
    # Add markers for each location
    for idx, row in df_agg.iterrows():
        # Calculate radius based on metric value
        radius = (row[metric_col] / max_val) * 20 + 5
        
        # Create popup text with all metrics
        popup_html = f"""
        <div style="font-family: Arial; min-width: 200px;">
            <h4 style="margin: 0 0 10px 0; color: #d62728;">{row['OutletPostCode']}</h4>
            <table style="width: 100%; font-size: 12px;">
                <tr><td><b>Total Cost Price:</b></td><td>£{row['Total Cost Price']:,.2f}</td></tr>
                <tr><td><b>Price Paid:</b></td><td>£{row['Price Paid']:,.2f}</td></tr>
                <tr><td><b>Product Cost:</b></td><td>£{row['Product Cost Price']:,.2f}</td></tr>
                <tr><td><b>Discount:</b></td><td>£{row['Discount']:,.2f}</td></tr>
                <tr><td><b>Duty Element:</b></td><td>£{row['Duty Element']:,.2f}</td></tr>
                <tr><td><b>Quantity:</b></td><td>{row['Qty']:,.0f}</td></tr>
            </table>
        </div>
        """
        
        # Color based on value (gradient from yellow to red)
        color_intensity = (row[metric_col] - min_val) / (max_val - min_val) if max_val != min_val else 0.5
        color = f'#{int(255):02x}{int(255 - color_intensity * 255):02x}00'
        
        # Add circle marker
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['OutletPostCode']}: £{row[metric_col]:,.2f}" if metric_col != 'Qty' else f"{row['OutletPostCode']}: {row[metric_col]:,.0f}",
            color='black',
            weight=1,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add a legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 180px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
        <p style="margin: 0 0 5px 0;"><b>{metric_col}</b></p>
        <p style="margin: 0; font-size: 12px;">Min: £{min_val:,.0f}</p>
        <p style="margin: 0; font-size: 12px;">Max: £{max_val:,.0f}</p>
        <p style="margin: 10px 0 0 0; font-size: 11px;">Circle size & color intensity represent value</p>
    </div>
    '''

    
    # Save map
    m.save(filename)
    print(f"Map saved: {filename}")
    return m

# Create maps for each metric
for col in key_cols:
    create_sales_map(df_agg, col, f'lancaster_map_{col.replace(" ", "_").lower()}.html')

# Create a combined heat map
m_heat = folium.Map(
    location=[54.0470, -2.8010],
    zoom_start=8,
    tiles='OpenStreetMap'
)

# Prepare heat map data (lat, lon, weight)
heat_data = [[row['lat'], row['lon'], row['Total Cost Price']] for idx, row in df_agg.iterrows()]

# Add heatmap layer
plugins.HeatMap(heat_data, 
                radius=15, 
                blur=25, 
                max_zoom=13,
                gradient={0.4: 'blue', 0.6: 'lime', 0.8: 'orange', 1.0: 'red'}).add_to(m_heat)

m_heat.save('lancaster_heatmap.html')
print("Heatmap saved: lancaster_heatmap.html")

print("\nSummary by Postcode")
print(df_agg[['OutletPostCode'] + key_cols].sort_values('Total Cost Price', ascending=False).head(10))
print(f"\nTotal locations: {len(df_agg)}")
print(f"\nOverall totals:")
for col in key_cols:
    if col != 'Qty':
        print(f"{col}: £{df_agg[col].sum():,.2f}")
    else:
        print(f"{col}: {df_agg[col].sum():,.0f}")


