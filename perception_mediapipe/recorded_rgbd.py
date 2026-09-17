"""Indexed read of RealSense ROS2 recordings; still uses rs.align for geometry.

Avoid the SDK player's repeated database searches across unrelated frames.
Software-device API: RealSense's align-with-software-device.py (Apache 2.0).
Only offline I/O changes. Live capture continues using D455Capture unchanged.
"""
import sqlite3
import numpy as np
import pyrealsense2 as rs
from rosbags.typesys import Stores,get_typestore


def fields(text):
    return dict(item.split('=',1) for item in text.split(';') if '=' in item)


class RecordedRGBD:
    def __init__(self,path):
        self.db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
        self.types=get_typestore(Stores.ROS2_HUMBLE)
        self.topics={name:identifier for identifier,name in self.db.execute('select id,name from topics')}
        self.prefix={'depth':'/device_0/sensor_0/Depth_0','color':'/device_0/sensor_1/Color_0'}
        self.meta={};self.streams={};self.sensors={};self.device=rs.software_device()
        self.device.create_matcher(rs.matchers.default)
        self.sync=rs.syncer(10);self.align=rs.align(rs.stream.color);self.sequence=0
        for uid,kind in enumerate(['depth','color'],1):
            prefix=self.prefix[kind];tid=self.topics[prefix+'/image/metadata']
            values=[]
            for stamp,blob in self.db.execute('select timestamp,data from messages where topic_id=? order by timestamp',(tid,)):
                v=fields(self.types.deserialize_cdr(blob,'std_msgs/msg/String').data)
                values.append(dict(stamp=stamp,frame=int(v['Frame number']),time=float(v['timestamp'])/1000,
                                   scale=float(v.get('depth_units',.001))))
            self.meta[kind]=values
            intr=fields(self.text(prefix+'/camera_info'));k=rs.intrinsics()
            for n in ['width','height']:setattr(k,n,int(intr[n]))
            for n in ['fx','fy','ppx','ppy']:setattr(k,n,float(intr[n]))
            k.coeffs=[float(x)for x in intr['coeffs'].split(',')]
            k.model=rs.distortion.brown_conrady if kind=='depth' else rs.distortion.inverse_brown_conrady
            v=rs.video_stream();v.type=rs.stream.depth if kind=='depth' else rs.stream.color
            v.width=k.width;v.height=k.height;v.fps=30;v.bpp=2 if kind=='depth' else 3
            v.fmt=rs.format.z16 if kind=='depth' else rs.format.rgb8;v.intrinsics=k;v.index=0;v.uid=uid
            sensor=self.device.add_sensor(kind);profile=sensor.add_video_stream(v)
            self.sensors[kind]=sensor;self.streams[kind]=profile
        ext=fields(self.text(self.prefix['color']+'/tf/ref_0'));e=rs.extrinsics()
        e.rotation=[float(x)for x in ext['rotation'].split(',')]
        e.translation=[float(x)for x in ext['translation'].split(',')]
        self.streams['depth'].register_extrinsics_to(self.streams['color'],e)
        for kind,sensor in self.sensors.items():sensor.open(self.streams[kind]);sensor.start(self.sync)
        self.colors={x['frame']:x for x in self.meta['color']}
        self.depth_times=np.array([x['time']for x in self.meta['depth']])
        k=self.streams['color'].as_video_stream_profile().intrinsics
        self.intr={n:getattr(k,n)for n in ['width','height','fx','fy','ppx','ppy']}
        self.intr.update(coeffs=list(k.coeffs),distortion_model=str(k.model))
        self.scale=self.meta['depth'][0]['scale']

    def text(self,name):
        blob=self.db.execute('select data from messages where topic_id=? limit 1',(self.topics[name],)).fetchone()[0]
        return self.types.deserialize_cdr(blob,'std_msgs/msg/String').data

    def read(self,frame_number,timestamp):
        c=self.colors.get(frame_number)
        if c is None or abs(c['time']-timestamp)>.001:raise ValueError('EXACT_SOURCE_FRAME_MISSING')
        i=int(np.argmin(abs(self.depth_times-c['time'])));d=self.meta['depth'][i]
        if abs(d['time']-c['time'])>1/60:raise ValueError('PAIRED_DEPTH_MISSING')
        self.sequence+=1;arrays={}
        # Synthetic packaging time is used only to feed these already matched
        # arrays to rs.align; all original source clocks are returned below.
        for kind,meta in [('depth',d),('color',c)]:
            tid=self.topics[self.prefix[kind]+'/image/data']
            blob=self.db.execute('select data from messages where timestamp=? and topic_id=?',(meta['stamp'],tid)).fetchone()[0]
            image=self.types.deserialize_cdr(blob,'sensor_msgs/msg/Image')
            if kind=='depth':array=image.data.view('<u2').reshape(image.height,image.width).copy()
            else:array=image.data.reshape(image.height,image.width,3).copy()
            arrays[kind]=array
            f=rs.software_video_frame();f.bpp=2 if kind=='depth' else 3;f.stride=image.width*f.bpp
            f.timestamp=self.sequence*1000/30;f.frame_number=self.sequence
            f.domain=rs.timestamp_domain.hardware_clock;f.pixels=array
            f.profile=self.streams[kind].as_video_stream_profile()
            if kind=='depth':f.depth_units=self.scale
            self.sensors[kind].on_video_frame(f)
        for _ in range(4):
            frame=self.sync.wait_for_frames(1000)
            if frame.get_color_frame() and frame.get_depth_frame():break
        else:raise RuntimeError('SOFTWARE_ALIGNMENT_PAIR_MISSING')
        aligned=self.align.process(frame)
        depth=np.asanyarray(aligned.get_depth_frame().get_data()).copy()
        return arrays['color'],depth,dict(depth_pair_checked=True,rgb_depth_skew_ms=(c['time']-d['time'])*1000,depth_frame_number=d['frame'])

    def close(self):
        for s in self.sensors.values():s.stop();s.close()
        self.db.close()
