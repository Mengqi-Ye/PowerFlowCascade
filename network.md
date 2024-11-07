```python
# Read lines and get start/end points of each line
df_endpoints: start and end points of lines

| osm_id  | geometry | type   |
| 111     | (x1, y1) | start  |
| 111     | (x2, y2) | end    |
| 115     | (x3, y3) | start  |
| 115     | (x4, y4) | end    |
| ...     | ...      | ...    |

# Start from the first first start point
df_subs: substations, geometry polygon
for i in df_endpoints:
    if i.osm_id is not in df_iter:
        for j in df_subs:
            if start_point.geometry is in j.geometry:
                search the end_point of start_point.osm_id
                for j in df_subs:
                    if end_point.geometry is in 
                search the start_point with the same end_point.geometry


df_subs: substations, geometry polygon
for i in df_endpoints:
    # Determine if a endpoint is inside a substation
    for j in df_subs:
        if i.geometry is in j.geometry:
            if i.type=='start':
                i.df_endpoints['sub']=j.osm_id
                i.df_endpoints['sub_type']=='from_sub'
            elif i.type=='end':
                i.df_endpoints['sub']=j.osm_id
                i.df_endpoints['sub_type']=='to_sub'
        else:
            i.df_endpoints['sub']=='none'
            i.df_endpoints['sub_type']=='none'

    #updated df_endpoints:
    | osm_id  | geometry | type   | sub    | sub_type |
    | line_id | (x1, y1) | start  | sub_id | from_sub |
    | line_id | (x2, y2) | end    | sub_id | to_sub   |
```

```python
df_iter = []
for i in df_endpoints:
    if i.sub != 'none':


```
![alt text](image.png)
在设置neighbour_threshold时，过小的话长的line上的点不会被分为neighbouring；过大的话，上图中这种并列的线上的点又会被分为一组。