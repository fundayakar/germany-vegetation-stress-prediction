//--------------------------------------
// Project: Vegetation stress prediction from antecedent hydroclimatic conditions
// -------------------------------------

// 1 - study area: entire Germany, adm1 list
var germany = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM0_NAME', 'Germany'));

// adm1 names for check
// print(germany.aggregate_array('ADM1_NAME').sort());


//  2 - land cover: ESA WorldCover 2020

var lc = ee.ImageCollection('ESA/WorldCover/v100').first();

var lcSimple = lc.remap(
  [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
  [ 1,  4,  3,  2,  4,  4,  4,  4,  4,  1,  4]
).rename('lc_class');
// 1=forest, 2=cropland, 3=grassland, 4=other

// 3 — NDVI + ERA5 collections, 2017-2024

var startDate = '2017-01-01';
var endDate   = '2024-12-31';

var modis = ee.ImageCollection('MODIS/061/MOD13A3')
  .filterDate(startDate, endDate)
  .filterBounds(germany)
  .select('NDVI')
  .map(function(img) {
    return img.multiply(0.0001).rename('ndvi')
      .copyProperties(img, ['system:time_start']);
  });

var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
  .filterDate(startDate, endDate)
  .filterBounds(germany)
  .select([
    'temperature_2m',
    'volumetric_soil_water_layer_1',
    'total_precipitation_sum'
  ])
  .map(function(img) {
    var temp = img.select('temperature_2m').subtract(273.15).rename('temp_c');
    var sm   = img.select('volumetric_soil_water_layer_1').rename('soil_moisture');
    var pr   = img.select('total_precipitation_sum')
                  .multiply(1000).rename('precip_mm');  // m → mm
    return temp.addBands(sm).addBands(pr)
      .copyProperties(img, ['system:time_start']);
  });

// 4 - Zonal stats cycle: year x month x ADM1 x LC
var years  = ee.List.sequence(2017, 2024);
var months = ee.List.sequence(1, 12);
var lcClasses = [1, 2, 3, 4];
var lcNames   = ['forest', 'cropland', 'grassland', 'other'];

var results = years.map(function(yr) {
  yr = ee.Number(yr);

  return months.map(function(mo) {
    mo = ee.Number(mo);
    var t0 = ee.Date.fromYMD(yr, mo, 1);
    var t1 = t0.advance(1, 'month');

    var ndviImg = modis.filterDate(t0, t1).mean().addBands(lcSimple);
    var climImg = era5.filterDate(t0, t1).mean();
    var combined = ndviImg.addBands(climImg);

    // adm1 × lc çift döngüsü
    var adm1Features = germany.toList(germany.size());

    return adm1Features.map(function(feat) {
      feat = ee.Feature(feat);
      var adm1Name = feat.get('ADM1_NAME');
      var geom     = feat.geometry();

      return ee.List(lcClasses).map(function(cls) {
        cls = ee.Number(cls);
        var mask   = lcSimple.eq(cls);
        var masked = combined.updateMask(mask);

        var stats = masked.reduceRegion({
          reducer: ee.Reducer.mean(),
          geometry: geom,
          scale: 1000,
          maxPixels: 1e9,
          bestEffort: true   // memory güvenliği için
        });

        return ee.Feature(null, stats
          .set('year',     yr)
          .set('month',    mo)
          .set('adm1',     adm1Name)
          .set('lc_class', cls)
          .set('lc_name',  ee.List(lcNames).get(cls.subtract(1)))
        );
      });
    });
  });
});

var flat = ee.FeatureCollection(results.flatten());

// 5 - Export

Export.table.toDrive({
  collection: flat,
  description: 'Germany_2017_2024_NDVI_ERA5_LC_monthly',
  fileFormat: 'CSV',
  selectors: [
    'year', 'month', 'adm1', 'lc_class', 'lc_name',
    'ndvi', 'temp_c', 'soil_moisture', 'precip_mm'
  ]
});
