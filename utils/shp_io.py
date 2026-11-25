import os

from osgeo import gdal, ogr, osr

__all__=['ShpIO']

class ShpIO():
    def __init__(self, filepath=None, update=False):
        ogr.RegisterAll()
        self._ds = None
        self._layer = None
        self._update=update
        self._geo_trans=None
        if filepath is not None:
            self.open(filepath, update=update)

    def open(self, filepath, update=False):
        if update:
            self._ds = ogr.Open(filepath, gdal.GA_Update)
        else:
            self._ds = ogr.Open(filepath)
        self._layer=self._ds.GetLayer(0)

        self._layer.ResetReading()

    @staticmethod
    def create(filepath,spatial=None):
        if os.path.exists(filepath):
            os.remove(filepath)
        ds=ogr.Driver.CreateDataSource(filepath)
        layer=ds.CreateLayer("Poly",spatial,ogr.wkbUnknown)
        return ShpIO(filepath,True)

    def create_field(self,field_name,type):
        if self._update:
            oField=ogr.FieldDefn(field_name,type)
            if self._layer.CreateField(oField,1)!=ogr.OGRERR_NONE:
                raise Exception("can't create field")

    def create_field_from_parent(self,layer):
        if self._update:
            layer_defn=layer.GetLayerDefn()
            for i in range(layer_defn.GetGetFieldCount()):
                field_defn=layer_defn.GetFieldDefn(i)
                field_name=field_defn.GetNameRef()
                type=field_defn.GetType()
                self.create_field(field_name,type)

    def create_fields(self,field_dict):
        if self._update:
            for key,value in field_dict.items():
                self.create(key,value)

    def get_layer(self):
        return self._layer

    def set_geo_trans(self,geo_trans):
        self._geo_trans=geo_trans

    def proj_geo_to_image(self, pts_geo):
        """
        geo coords in [x, y] to [cols, rows] in image projection
        """
        assert self._geo_trans!=None
        tInv = gdal.InvGeoTransform(self._geo_trans)
        pts = [gdal.ApplyGeoTransform(tInv, p[0], p[1]) for p in pts_geo]
        return pts

    def proj_image_to_geo(self, pts):
        """
        [cols, rows] -> [x, y]
        """
        assert self._geo_trans != None
        t = self._geo_trans
        pts_geo = [gdal.ApplyGeoTransform(t, p[0], p[1]) for p in pts]
        return pts_geo

    # def proj_geo_to_image(self,geo_x,geo_y):
    #     if self._geo_trans is None:
    #         raise Exception("geotrans is None")
    #
    #     geoTrans=self._geo_trans
    #     dTemp = geoTrans[1] * geoTrans[5] - geoTrans[2] * geoTrans[4]
    #     Xp=geo_x
    #     Yp=geo_y
    #     Xpixel = (geoTrans[5] * (Xp - geoTrans[0]) - geoTrans[2] * (Yp - geoTrans[3])) / dTemp + 0.5
    #     Yline = (geoTrans[1] * (Yp - geoTrans[3]) - geoTrans[4] * (Xp - geoTrans[0])) / dTemp + 0.5
    #     return Xpixel,Yline

    def get_next_feature(self):
        return self._layer.GetNextFeature()

    def get_feature(self,i):
        if i > self._layer.GetFeatureCount() or i < 0:
            raise Exception("index out of the layer count")
        return self._layer.GetFeature(i)

    def get_points_from_featureindex(self,i):
        feature=self._layer.GetFeature(i)
        return self.get_points_from_feature(feature)

    def get_points_from_feature(self,feature:ogr.Feature):
        """

        Args:
            feature:

        Returns:
            返回feature中的所有顶点，[[ring1_points],[ring2_points]]
        """
        points = []
        rings = []
        geometry = feature.GetGeometryRef()
        if geometry is None:
            return points

        for i in range(geometry.GetGeometryCount()):
            rings.append(geometry.GetGeometryRef(i))

        for ring in rings:
            _ = []
            nVertices = ring.GetPointCount()
            #最后一个点和第一个点相同，可以去掉
            for i in range(0, nVertices-1):
                point = ring.GetPoint(i)
                _.append(point)
            points.append(_)

        return points

    def transformer_points(self,feature_index,geo_trans=None):
        if geo_trans is not None:
            self.set_geo_trans(geo_trans)
        points=self.get_points_from_feature(feature_index)
        for ring_points in points:
            for point in ring_points:
                x,y=self.proj_geo_to_image([point.x,point.y])

    def get_id_affineparam_dict(self,id_name:str):
        """
        从图层中获取放射参数
        Returns:

        """
        self._layer.ResetReading()
        id_filed_dict={}
        feature=self._layer.GetNextFeature()

        while feature is not None:
            # if feature.GetFieldIndex("connectID")!=1:
            #     print("dont exist field 'connectID'")
            #     break
            building_id=feature.GetFieldAsString(id_name)
            a=feature.GetFieldAsDouble("a")
            b=feature.GetFieldAsDouble("b")
            c=feature.GetFieldAsDouble("c")
            d=feature.GetFieldAsDouble("d")
            e=feature.GetFieldAsDouble("e")
            f=feature.GetFieldAsDouble("f")

            id_filed_dict[building_id]=[a,b,c,d,e,f]

            feature=self._layer.GetNextFeature()

        self._layer.ResetReading()
        return id_filed_dict

    def get_features_by_field_index(self,field:str,value):

        assert self._layer.FindFieldIndex(field,False)!=-1
        self._layer.ResetReading()
        out_features=[]
        feature=self._layer.GetNextFeature()
        while feature!=None:
            field_value=feature.GetFieldAsString(field)
            if field_value==str(value):
                out_features.append(feature)
            feature=self._layer.GetNextFeature()

        self._layer.ResetReading()
        return out_features

    def get_points_from_field_index(self,field:str,value):
        features=self.get_features_by_field_index(field,value)
        out_points=[]
        for feature in features:
            points=self.get_points_from_feature(feature)
            out_points.append(points)
        return out_points
