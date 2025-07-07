# add agent
# track_infos["object_id"]=np.concatenate([track_infos["object_id"],np.zeros([1])])
# track_infos["valid"]=np.concatenate([track_infos["valid"],np.ones([1,91]).astype(bool)])
# track_infos["role"]=np.concatenate([track_infos["role"],np.zeros([1,3]).astype(bool)])

# add static object
# track_infos["object_type"]=np.concatenate([track_infos["object_type"],np.zeros([1]).astype(int)])
# state=np.zeros([1,91,9])# x, y, z, length, width, height,heading,vx,vy
# state[0,:,0]=362
# state[0,:,1]=6300
# state[0,:,3]=10
# state[0,:,4]=10
# state[0,:,5]=10
# track_infos["states"]=np.concatenate([track_infos["states"],state])

# add dynamic ped
# track_infos["object_type"]=np.concatenate([track_infos["object_type"],np.ones([1]).astype(int)])
# state=np.zeros([1,91,9])# x, y, z, length, width, height,heading,vx,vy
#
# state[0,0,0]=362
# state[0,5,0]=361
# state[0,10,0]=360
# state[0,:,1]=6270
# state[0,:,3]=1
# state[0,:,4]=1
# state[0,:,5]=1
# state[0,:,6]=-np.pi
# track_infos["states"]=np.concatenate([track_infos["states"],state])


# delete agent
# id=997
# mask=np.where(track_infos["object_id"]!=id)
# track_infos["object_id"]=track_infos["object_id"][mask]
# track_infos["object_type"]=track_infos["object_type"][mask]
# track_infos["states"]=track_infos["states"][mask]
# track_infos["valid"]=track_infos["valid"][mask]
# track_infos["role"]=track_infos["role"][mask]
