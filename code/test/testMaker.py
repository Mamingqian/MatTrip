import os
import folium
from folium.features import CustomIcon

from folium import FeatureGroup, LayerControl, Map, Marker

from folium.plugins import MarkerCluster


m = folium.Map(location=[44, -73], zoom_start=5)

marker_cluster = MarkerCluster().add_to(m)


folium.Marker(
    location=[40.67, -73.94],
    popup='Add popup text here.',
    icon=folium.Icon(color='green', icon='ok'),
).add_to(marker_cluster)

folium.Marker(
    location=[44.67, -73.94],
    popup='Add popup text here.',
    icon=folium.Icon(color='red'),
).add_to(marker_cluster)

folium.Marker(
    location=[44.67, -71.94],
    popup='Add popup text here.',
    icon=None,
).add_to(marker_cluster)







# whiteMarker = folium.features.CustomIcon("file:///Users/cxyang/Documents/TourRec%20Code/marker/WhiteMarker.png", icon_size=(95, 95))

# m = Map(
#     location=[45.372, -121.6972],
#     zoom_start=12,
#     tiles='Stamen Terrain'
# )

# feature_group = FeatureGroup(name='Some icons')
# # Marker(location=[45.3288, -121.6625],
# #        popup='Mt. Hood Meadows', icon=whiteMarker).add_to(feature_group)

# Marker(location=[45.3311, -121.7113],
#        popup='Timberline Lodge', icon=whiteMarker).add_to(feature_group)

# feature_group.add_to(m)
# LayerControl().add_to(m)





# whiteMarker = folium.features.CustomIcon("file:///Users/cxyang/Documents/TourRec%20Code/marker/WhiteMarker.png", icon_size=(95, 95))
# m = folium.Map(location=[45.372, -121.6972], zoom_start=12, tiles='Stamen Terrain')


# marker1 = folium.Marker(
#     location=[45.3288, -121.6625],
#     icon=whiteMarker,
#     popup='Mt. Hood Meadows'
# )

# marker2 = folium.Marker(
#     location=[45.3088, -121.6625],
#     icon=whiteMarker,
#     popup='Mt. Hood Meadows'
# )

# # marker = folium.Marker(
# #     location=[45.3088, -121.7125],
# #     icon=whiteMarker,
# #     popup='Mt. Hood Meadows'
# # ).add_to(m)

# m.add_child(marker1)
# m.add_child(marker2)

m.save("result.html")