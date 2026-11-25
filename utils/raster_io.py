# raster io, a wraper of gdal
# import os
# import sys
# import random
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from osgeo import gdal, ogr, osr
from osgeo import gdal, ogr, osr
import numpy as np

OPTIONS = ("COMPRESS=DEFLATE",)
__all__ = ['convert_uint16_to_uint8', 'RasterIO', 'fastly_save_image', 'OPTIONS']


# def write_chw_array_to_image(file_path, im):
#     cv2.imwrite(file_path, np.transpose(im[...,::-1], (1,2,0)))

# def resize_prob_chw(prob, shape):
#     '''
#         resize prob in log space
#         input: prob: numpy array, chw format, dtype = np.uint8
#         example downsize to half:
#         a = (np.zeros((3,10,10),dtype=np.uint8) + 255)/3
#         resized_prob = resize_prob_chw(a, a.shape[0:2]/2)
#     '''
#
#     assert prob.dtype == np.uint8
#
#     prob_log = np.log(prob.astype(np.float32)/255)
#
#     # remove floating point erros
#     prob_log[np.isinf(prob_log)] = -20
#
#     prob_log = cv2.resize(prob_log, shape)
#
#     prob = np.exp(prob_log)
#     prob = prob / np.sum(prob, axis = 0) * 255
#
#     # remove potential extrem errors that may be caused by computing precision
#     prob[prob > 255] = 255
#     prob[prob < 0] = 0
#     prob = prob.astype(np.uint8)
#
#     return prob

def convert_uint16_to_uint8(data, level=2):
    for i in range(data.shape[0]):
        a = np.percentile(data[i], level)
        b = np.percentile(data[i], 100 - level)
        e = 255 * (data[i] - a) / (b - a)
        e[e < 0] = 0
        e[e > 255] = 255
        data[i] = e.astype(np.uint8)
    return data.astype(np.uint8)


class RasterIO:
    datatype={"1":"Byte",
              "2":"UInt16",
              "3":"Int16",
              "4":"UInt32",
              "5":"Int32",
              "6":"Float32",
              "7":"Float64",
              "8":"CInt16",
              "9":"CInt32",
              "10":"CFloat32",
              "11":"CFloat64"}

    gdal_data_type={"1":gdal.GDT_Byte,
              "2":gdal.GDT_UInt16,
              "3":gdal.GDT_Int16,
              "4":gdal.GDT_UInt32,
              "5":gdal.GDT_Int32,
              "6":gdal.GDT_Float32,
              "7":gdal.GDT_Float64,
              "8":gdal.GDT_CInt16,
              "9":gdal.GDT_CInt32,
              "10":gdal.GDT_CFloat32,
              "11":gdal.GDT_CFloat64}

    def __init__(self, filepath=None, file=None, update=False):
        self._ds = file
        self.updata = update
        if filepath is not None:
            self.open(filepath, update=update)
        # elif file is not None:

    def open(self, filepath, update=False):
        if update:
            self._ds = gdal.Open(filepath, gdal.GA_Update)
        else:
            self._ds = gdal.Open(filepath)
        self.update=update
        trans = self.get_geo_transform(self._ds)
        if trans != self._ds.GetGeoTransform() and self.update:
            self._ds.SetGeoTransform(trans)
            
        self.img_w=self._ds.RasterXSize
        self.img_h=self._ds.RasterYSize
        
    def open_and_update(self, filepath):
        self._ds = gdal.Open(filepath, gdal.GA_Update)
        trans = self.get_geo_transform(self._ds)
        if trans != self._ds.GetGeoTransform():
            self._ds.SetGeoTransform(trans)

    def create(self, filepath, rows, cols, channels, options=(), gdal_type=gdal.GDT_Byte):
        """ example options:
        options=["TILED=YES", "COMPRESS=PACKBITS"]
        options=["TILED=YES", "COMPRESS=JPEG"]
        options=["TILED=YES", "COMPRESS=JPEG", "JPEG_QUALITY=85"]
        options=["TILED=YES", "COMPRESS=DEFLATE", "GDAL_TIFF_INTERNAL_MASK=YES"]
        """
        driver = gdal.GetDriverByName("GTiff")
        self._ds = driver.Create(filepath,
                                 cols,
                                 rows,
                                 channels,
                                 gdal_type,
                                 options=options)
        self.updata=True
        self.img_w=cols
        self.img_h=rows
        
    def read_as_array(self):
        return self._ds.ReadAsArray().astype(np.float32)

    def read_as_array_with_scale(self,scale):
        shape=self.shape()[1:]
        return self._ds.ReadAsArray(xoff=0,yoff=0,xsize=shape[1],ysize=shape[0],
                                    buf_xsize=int(shape[1]*scale),buf_ysize=int(shape[0]*scale)).astype(np.float32)
      
    def read_as_given_hw(self,h,w):
        shape=self.shape()[1:]
        return self._ds.ReadAsArray(xoff=0,yoff=0,xsize=shape[1],ysize=shape[0],
                                    buf_xsize=w,buf_ysize=h).astype(np.float32)
    
    def read_as_given_xyhw(self,offset_x,offset_y,h,w):
        return self._ds.ReadAsArray(xoff=offset_x,yoff=offset_y,xsize=w,ysize=h,
                                    buf_xsize=w,buf_ysize=h).astype(np.float32)
    
    def read_as_given_xyhw_resize(self,offset_x,offset_y,h,w,resize_y,resize_w):
        return self._ds.ReadAsArray(xoff=offset_x,yoff=offset_y,xsize=w,ysize=h,
                                    buf_xsize=resize_w,buf_ysize=resize_y).astype(np.float32)
    
    def read_band(self, band):
        return self._ds.GetRasterBand(band + 1).ReadAsArray()

    def get_band(self,band):
        return self._ds.GetRasterBand(band + 1)

    def shape(self):
        """
        Returns:
            图像的形状------>[C,H,W]
        """
        return [self._ds.RasterCount, self._ds.RasterYSize, self._ds.RasterXSize]

    def get_data_type(self):
        return self.datatype[str(self._ds.GetRasterBand(1).DataType)]

    def get_gdal_data_type(self):
        return self.gdal_data_type[str(self._ds.GetRasterBand(1).DataType)]
    
    def write_band(self, data, channel):
        if self.updata:
            self._ds.GetRasterBand(channel + 1).WriteArray(data)
            # self._ds.GetRasterBand(channel + 1).WriteRaster(0,0,data.shape[0],data.shape[1],buf_ysize=self.img_h,buf_xsize=self.img_w)
    def write_array(self, data):
        if self.updata:
            # self._ds.WriteRaster(0,0,data.shape[1],data.shape[2],buf_ysize=self.img_h,buf_xsize=self.img_w)
            for i, d in enumerate(data):
                self._ds.GetRasterBand(i + 1).WriteArray(d)
            self._ds.FlushCache()

    @staticmethod
    def get_geo_transform(ds):
        geo_trans = ds.GetGeoTransform()
        if geo_trans == (0, 1, 0, 0, 0, 1):
            # print('flip y axis of image to downwards, translation: (' +
            #       ', '.join([str(g) for g in geo_trans]) + ')')
            geo_trans = (0, 1, 0, 0, 0, -1)
        return geo_trans

    @staticmethod
    def get_geo_tansform_from_file(file):
        ds=gdal.Open(file)
        return ds.GetGeoTransform()

    def copy_projection(self, filepath, location=None):
        ds = gdal.Open(filepath)
        geo_trans = self.get_geo_transform(ds)
        if location:
            geo_trans = list(geo_trans)
            geo_trans[0] = geo_trans[0] + geo_trans[1] * location.coff + geo_trans[2] * location.roff
            geo_trans[3] = geo_trans[3] + geo_trans[4] * location.coff + geo_trans[5] * location.roff
        prj = ds.GetProjectionRef()
        # if prj == '':
        #    srs = osr.SpatialReference()
        #    srs.ImportFromEPSG(4326)
        #    prj = srs.ExportToWkt()
        self._ds.SetProjection(prj)
        self._ds.SetGeoTransform(geo_trans)
        self._ds.FlushCache()

    def proj_geo_to_image(self, pts_geo):
        """
        geo coords in [x, y] to [cols, rows] in image projection
        """
        tInv = gdal.InvGeoTransform(self.get_geo_transform(self._ds))
        pts = [gdal.ApplyGeoTransform(tInv, p[0], p[1]) for p in pts_geo]
        return pts

    def proj_image_to_geo(self, pts):
        """
        [cols, rows] -> [x, y]
        """
        t = self._ds.GetGeoTransform()
        pts_geo = [gdal.ApplyGeoTransform(t, p[0], p[1]) for p in pts]
        return pts_geo

    def proj_geometry_to_im(self, g, xoff=0, yoff=0):
        def proj_linestring_ring(g):
            pts = self.proj_geo_to_image(g.GetPoints())
            for j, p in enumerate(pts):
                g.SetPoint(j, p[0] - xoff, p[1] - yoff)

        if g.GetGeometryName() == 'POLYGON' or g.GetGeometryName() == 'MULTILINESTRING':
            for i in range(g.GetGeometryCount()):
                ring = g.GetGeometryRef(i)
                proj_linestring_ring(ring)
                # pts = self.proj_geo_to_image(ring.GetPoints())
                # for j, p in enumerate(pts):
                #    ring.SetPoint(j, p[0] - xoff, p[1] - yoff)
        elif g.GetGeometryName() == 'LINESTRING':
            proj_linestring_ring(g)
            # pts = self.proj_geo_to_image(g.GetPoints())
            # for j, p in enumerate(pts):
            #    g.SetPoint(j, p[0] - xoff, p[1] - yoff)
        elif g.GetGeometryName() == 'MULTIPOLYGON':
            for i in range(g.GetGeometryCount()):
                poly = g.GetGeometryRef(i)
                for j in range(poly.GetGeometryCount()):
                    ring = poly.GetGeometryRef(j)
                    proj_linestring_ring(ring)
        else:
            print(g.GetGeometryName())
            raise "Not Implemented"

    def proj_geo_envelop_to_image(self, env):
        # env: (minx, maxx, miny, maxy)
        # env is derectly generated from geometry.GetEnvelop()
        pts_geo = [[env[0], env[3]], [env[1], env[3]],
                   [env[1], env[2]], [env[0], env[2]]]
        pts = self.proj_geo_to_image(pts_geo)
        return pts

    def proj_geo_envelop_to_image_envelop(self, env):
        pts = self.proj_geo_envelop_to_image(env)
        x = [i[0] for i in pts]
        y = [i[1] for i in pts]
        return (min(x), max(x), min(y), max(y))

    def set_geo_transform(self, geo_trans):
        if self.updata:
            self._ds.SetGeoTransform(geo_trans)
            self._ds.FlushCache()

    def get_geo_trans(self):
        return self._ds.GetGeoTransform()

    def set_none_data(self,none_data=0):
        if self.updata:
            for i in range(0,self._ds.RasterCount):
                band=self._ds.GetRasterBand(i+1)
                band.SetNoDataValue(none_data)

            

def fastly_save_image(image_data: np.ndarray, out_file, ref_file=None,
                      geo_trans=None, options=OPTIONS, gdal_type=gdal.GDT_Byte,buf_xsize=None,buf_ysize=None):
    if options is None:
        options = ()
    if ref_file is not None and geo_trans is not None:
        raise Exception('ref_file and geo_trans')
    if image_data.ndim == 2:
        fastly_save_image_2dim(image_data, out_file, ref_file, geo_trans, options, gdal_type,buf_xsize,buf_ysize)
    elif image_data.ndim == 3:
        fastly_save_image_3dim(image_data, out_file, ref_file, geo_trans, options, gdal_type,buf_xsize,buf_ysize)
    else:
        raise Exception('image_data.ndim is not 2 or 3 !')


def fastly_save_image_2dim(image_data: np.ndarray, out_file, ref_file,
                           geo_trans=None, options=OPTIONS, gdal_type=gdal.GDT_Byte,buf_xsize=None,buf_ysize=None):
    if options is None:
        options = ()
    if image_data.ndim != 2:
        raise Exception('image_data.ndim != 2')
    io = RasterIO()
    # io.create(out_file,image_data.shape[0], image_data.shape[1], 1, options, gdal_type)
    io.create(out_file,buf_ysize if buf_ysize else image_data.shape[0], 
              buf_xsize if buf_xsize else image_data.shape[1], 1, options, gdal_type)
    io.set_none_data()
    io.write_band(image_data, 0)
    if geo_trans is None and ref_file is not None:
        geo_trans=RasterIO.get_geo_tansform_from_file(ref_file)

    if geo_trans is not None:
        io.set_geo_transform(geo_trans)
    if ref_file is not None:
        io.copy_projection(ref_file)
    del io


def fastly_save_image_3dim(image_data: np.ndarray, out_file, ref_file,
                           geo_trans=None, options=OPTIONS, gdal_type=gdal.GDT_Byte,buf_xsize=None,buf_ysize=None):
    if options is None:
        options = ()
    if image_data.ndim != 3:
        raise Exception('image_data.ndim != 3')
    io = RasterIO()
    # io.create(out_file, image_data.shape[1], image_data.shape[2], image_data.shape[0], options, gdal_type)
    io.create(out_file,buf_ysize if buf_ysize else image_data.shape[1], 
            buf_xsize if buf_xsize else image_data.shape[2], image_data.shape[0], options, gdal_type)
    
    # io.set_none_data()
    io.write_array(image_data)
    if geo_trans is None and ref_file is not None:
        geo_trans=RasterIO.get_geo_tansform_from_file(ref_file)
    if geo_trans is not None:
        io.set_geo_transform(geo_trans)
    if ref_file is not None:
        io.copy_projection(ref_file)
    del io
