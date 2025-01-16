#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <array>
#include <list>
#include <cmath>
#include <algorithm>
#include <limits>

// -------------
// pure C++ code
// -------------

using Coords = std::array<double, 3>;
using CoordsList = std::list<Coords>;

double distance( const Coords& coords1,
		 const Coords& coords2 )
{
  // Euclidean square root distance
  return sqrt( (coords1[0] - coords2[0]) * (coords1[0] - coords2[0]) +
               (coords1[1] - coords2[1]) * (coords1[1] - coords2[1]) );
}


std::pair<double, double> vector_statistics(const std::vector<double> vect)
{
  double sum = std::accumulate(vect.begin(), vect.end(), 0.0);
  double mean = sum / vect.size();
  std::vector<double> diff(vect.size());
  std::transform(vect.begin(), vect.end(), diff.begin(), [mean](double x) { return x - mean; });
  double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  return std::make_pair(mean, std::sqrt(sq_sum / vect.size()));
}


std::vector<long int> getNeighboringCells(const long int cellCol,
					  const long int cellRow,
					  const long int xSize,
					  const long int ySize,
					  const long int radius)
{

  std::vector<long int> neighboringCells;

  // window of 2 * radius + 1 centered on the current cell coordinates
  long int minCol = std::max<long int>(-radius, cellCol - radius);
  long int minRow = std::max<long int>(-radius, cellRow - radius);
  long int maxCol = std::min<long int>((xSize-1)+radius, cellCol + radius);
  long int maxRow = std::min<long int>((ySize-1)+radius, cellRow + radius);

  for(long int r = minRow; r <= maxRow; r++){
    for(long int c = minCol; c <= maxCol; c++){
      neighboringCells.push_back( (r+radius) * (xSize+2*radius) + (c+radius) );
    }
  }

  return neighboringCells;
}


typedef std::tuple<std::vector<double>, // gaussian interpolation
		   double, // sum of weights
		   std::vector<double>, // mean
		   std::vector<double>, // standard deviation
		   uint16_t, // nb points in discus
		   uint16_t  // nb points in cell
		   > gaussianType;

gaussianType getGaussian(const std::vector<double>& valuesVector,
			 const std::vector<int>& validVector,
			 const std::vector< CoordsList >& gridToInterpol,
			 const std::vector<long int>& neighbors,
			 const long int cellCol,
			 const long int cellRow,
			 const long int xSize,
			 const long int ySize,
			 const double sigma,
			 const long int radius,
			 const double resolution,
			 const long int nbBands,
			 const long int nbPoints)
{

  // We take the coordinates at the center of the pixel
  Coords cellCoordsCenter = Coords{ static_cast<double>(cellCol + 0.5f),
                                    static_cast<double>(cellRow + 0.5f),
                                    -1.f };
  std::list < std::pair<double, Coords> > coordsList;
  uint16_t nbPointsInCell = 0;
  std::vector<long int> indexes;
  std::vector<double> weights;
  std::vector<double> gaussian_interp(nbBands, std::numeric_limits<double>::quiet_NaN());
  std::vector<double> mean(nbBands, std::numeric_limits<double>::quiet_NaN());
  std::vector<double> stdev(nbBands, std::numeric_limits<double>::quiet_NaN());

  for(const auto neigh : neighbors){
    for(const auto& coords: gridToInterpol[neigh]){
      double dist = distance(cellCoordsCenter, coords);
      if (dist < 0.5 + radius) {
	dist *= resolution;
	weights.push_back(exp( (- (dist * dist) / (2 * sigma * sigma) )));
	indexes.push_back(coords[2]);

	if ((std::abs(coords[0] - (cellCol + 0.5f)) < 0.5f) &&
	    (std::abs(coords[1] - (cellRow + 0.5f)) < 0.5f)) {
	  nbPointsInCell++;
	}
      }
    }
  }

  bool noValidNeighbor = true;
  for (const auto index : indexes) {
    if (validVector[index] != 0) {
      noValidNeighbor = false;
      break;
    }
  }

  uint16_t nbPointsInDisc;
  double weightsSum = 0;

  if (noValidNeighbor == false) {
    nbPointsInDisc = indexes.size();
    weightsSum = std::accumulate(weights.begin(),
				 weights.end(),
				 0.0);

    if(nbPointsInDisc > 0){
      for( long int band = 0; band < nbBands ; ++band) {
	std::vector<double> indexesValue(nbPointsInDisc);
	gaussian_interp[band] = 0;
	for( long int point = 0; point < nbPointsInDisc ; ++point) {
	  double weight = weights[point];
	  long int index = indexes[point];
	  double value = valuesVector[band*nbPoints+index];
	  indexesValue[point] = value;
	  gaussian_interp[band] += weight*value;
	}
	gaussian_interp[band] /= weightsSum;

	auto [mean_, stdev_] = vector_statistics(indexesValue);
	mean[band] = mean_;
	stdev[band] = stdev_;
      }
    }

  }
  else {
    nbPointsInCell = 0;
    nbPointsInDisc = 0;
  }

  return std::make_tuple(gaussian_interp,
			 weightsSum,
			 mean, stdev,
			 nbPointsInDisc,
			 nbPointsInCell);
}

std::vector<float> pointCloudToDSM(const std::vector<double>& pointsVector,
				   const std::vector<double>& valuesVector,
				   const std::vector<int>& validVector,
				   const long int nbBands,
				   const long int nbPoints,
				   const double xStart,
				   const double yStart,
				   const long int xSize,
				   const long int ySize,
				   const double resolution,
				   const long int radius,
				   const double sigma,
				   std::vector<float>& weightsSumVector,
				   std::vector<float>& meanVector,
				   std::vector<float>& stdevVector,
				   std::vector<uint16_t>& nbPointsInDiscVector,
				   std::vector<uint16_t>& nbPointsInCellVector)
{
  const long int outSize = xSize*ySize;
  std::vector<float> outputVector(nbBands*outSize);
  weightsSumVector.resize(outSize);
  meanVector.resize(nbBands*outSize);
  stdevVector.resize(nbBands*outSize);
  nbPointsInDiscVector.resize(outSize);
  nbPointsInCellVector.resize(outSize);

  // For each target pixel, we store a list of fractional Z coordinates
  const long int outSizeWithMargins = (xSize+2*radius)*(ySize+2*radius);
  std::vector< CoordsList > gridToInterpolWithMargins(outSizeWithMargins);
  double x, y, col, row;
  long int cellCol, cellRow, rowByCol;

  for ( long int k = 0 ; k < nbPoints ; ++k ) {

    x = pointsVector[(0*nbPoints)+k];
    y = pointsVector[(1*nbPoints)+k];

    col = (x - xStart) / resolution;
    row = (yStart - y) / resolution;

    cellCol = floor(col);
    cellRow = floor(row);

    if ((cellCol >= -radius) && (cellCol < xSize+radius) \
	&& (cellRow >= -radius) && (cellRow < ySize+radius)) {
      rowByCol = (cellCol+radius) + (cellRow+radius) * (xSize+2*radius);
      gridToInterpolWithMargins[rowByCol].push_back(Coords{col, row, static_cast<double> (k)});
    }
  }

  // Loop over the grid to interpolate the z for each cell
  for ( long int k = 0 ; k < outSize ; ++k ) {

    cellCol = k % xSize;
    cellRow = k / xSize;

    // Get neighboring cells with radius defined by the user
    auto neighbors = getNeighboringCells(cellCol,
                                         cellRow,
                                         xSize,
                                         ySize,
                                         radius);

    // Gaussian interpolation
    auto [out,
	  weightsSum,
	  mean,
	  stdev,
	  nbPointsInDisc,
	  nbPointsInCell] = getGaussian(valuesVector,
					validVector,
					gridToInterpolWithMargins,
					neighbors,
					cellCol,
					cellRow,
					xSize,
					ySize,
					sigma,
					radius,
					resolution,
					nbBands,
					nbPoints);

    for( long int band = 0; band < nbBands ; ++band) {
      outputVector[k*nbBands+band] = out[band];
      meanVector[k*nbBands+band] = mean[band];
      stdevVector[k*nbBands+band] = stdev[band];
    }

    nbPointsInDiscVector[k] = nbPointsInDisc;
    nbPointsInCellVector[k] = nbPointsInCell;
    weightsSumVector[k] = weightsSum;
  }

  return outputVector;
}

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

// wrap C++ function with NumPy array IO
typedef std::tuple<py::array, // gaussian interpolation
		   py::array, // sum of weights
		   py::array, // mean
		   py::array, // standard deviation
		   py::array, // nb points in discus
		   py::array // nb points in cell
		   > pyPointCloudtoDSMType;

pyPointCloudtoDSMType pyPointCloudToDSM(py::array_t<double,
					py::array::c_style | py::array::forcecast> points,
					py::array_t<double,
					py::array::c_style | py::array::forcecast> values,
					py::array_t<int,
					py::array::c_style | py::array::forcecast> valid,
					double xStart,
					double yStart,
					size_t xSize,
					size_t ySize,
					double resolution,
					size_t radius,
					double sigma)
{
  // check input dimensions
  if ( points.ndim()     != 2 )
    throw std::runtime_error("points should be 2-D NumPy array");

  if ( points.shape()[0] != 2 )
    throw std::runtime_error("points should have size [2, N]");

  if ( values.ndim()     != 2 )
    throw std::runtime_error("values should be 2-D NumPy array");

  if ( values.shape()[1] != points.shape()[1] ) {
    std::string message = "values and points should have the same second dimension";
    message = message + std::string("(") + std::to_string(values.shape()[1]);
    message = message + std::string(" ~= ") + std::to_string(points.shape()[1]) + std::string(")");
    throw std::runtime_error(message);
  }

  if ( valid.ndim()     != 2 )
    throw std::runtime_error("valid should be 2-D NumPy array");

  if ( valid.shape()[1] != points.shape()[1] ) {
    std::string message = "valid and points should have the same second dimension";
    message = message + std::string("(") + std::to_string(valid.shape()[1]);
    message = message + std::string(" ~= ") + std::to_string(points.shape()[1]) + std::string(")");
    throw std::runtime_error(message);
  }

  size_t nbBands = values.shape()[0];
  size_t nbPoints = points.shape()[1];

  // allocate std::vector (to pass to the C++ function)
  std::vector<double> pointsVector(points.size());
  std::vector<double> valuesVector(values.size());
  std::vector<int> validVector(valid.size());

  // copy py::array -> std::vector
  // Do we need to make a copy here ? Is there something similar
  // to cython memory view
  // https://stackoverflow.com/questions/54793539/pybind11-modify-numpy-array-from-c
  // It seems that passing the py::array by reference should do the job if
  // there is not conflicting type, however the function pointCloudToDSM needs a double * ptr as
  // input instead of a std::vector.
  std::memcpy(pointsVector.data(),points.data(),points.size()*sizeof(double));
  std::memcpy(valuesVector.data(),values.data(),values.size()*sizeof(double));
  std::memcpy(validVector.data(),valid.data(),valid.size()*sizeof(int));

  std::vector<float> weightsSumVector;
  std::vector<float> meanVector;
  std::vector<float> stdevVector;
  std::vector<uint16_t> nbPointsInDiscVector;
  std::vector<uint16_t> nbPointsInCellVector;

  // call pure C++ function
  auto outputVector = pointCloudToDSM(pointsVector,
				      valuesVector,
				      validVector,
				      nbBands,
				      nbPoints,
				      xStart,
				      yStart,
				      xSize,
				      ySize,
				      resolution,
				      radius,
				      sigma,
				      weightsSumVector,
				      meanVector,
				      stdevVector,
				      nbPointsInDiscVector,
				      nbPointsInCellVector);

  // out, mean, stdev for each band + n_pts and n_in_cell
  size_t              ndim    = 3;
  std::vector<size_t> shape   = { xSize, ySize, nbBands };
  std::vector<size_t> strides = { nbBands*ySize*sizeof(float),
				  nbBands*sizeof(float),
				  sizeof(float)};

  // return 2-D NumPy array, I think here it is ok since the expected argument is
  // a pointer so there is no copy
  auto out = py::array(py::buffer_info(outputVector.data(),
				       sizeof(float),
				       py::format_descriptor<float>::format(),
				       ndim,
				       shape,
				       strides
				       ));

  auto mean = py::array(py::buffer_info(meanVector.data(),
					sizeof(float),
					py::format_descriptor<float>::format(),
					ndim,
					shape,
					strides
					));

  auto stdev = py::array(py::buffer_info(stdevVector.data(),
					 sizeof(float),
					 py::format_descriptor<float>::format(),
					 ndim,
					 shape,
					 strides
					 ));

  ndim    = 2;
  shape   = { xSize, ySize };
  strides = { ySize*sizeof(float),
	      sizeof(float)};

  auto weightsSum = py::array(py::buffer_info(weightsSumVector.data(),
					      sizeof(float),
					      py::format_descriptor<float>::format(),
					      ndim,
					      shape,
					      strides
					      ));

  ndim    = 2;
  shape   = { xSize, ySize };
  strides = { ySize*sizeof(uint16_t),
	      sizeof(uint16_t)};

  auto nbPointsInDisc = py::array(py::buffer_info(nbPointsInDiscVector.data(),
						  sizeof(uint16_t),
						  py::format_descriptor<uint16_t>::format(),
						  ndim,
						  shape,
						  strides
						  ));

  auto nbPointsInCell = py::array(py::buffer_info(nbPointsInCellVector.data(),
						  sizeof(uint16_t),
						  py::format_descriptor<uint16_t>::format(),
						  ndim,
						  shape,
						  strides
						  ));

  return std::make_tuple(out, weightsSum, mean, stdev, nbPointsInDisc, nbPointsInCell);

}

// wrap as Python module
PYBIND11_MODULE(rasterize, m)
{
  m.doc() = "rasterize";

  m.def("pc_to_dsm", &pyPointCloudToDSM, "Convert point cloud to digital surface model",
	py::arg("points"),
	py::arg("values"),
	py::arg("valid"),
	py::arg("xstart"),
	py::arg("ystart"),
	py::arg("xsize"),
	py::arg("ysize"),
	py::arg("resolution"),
	py::arg("radius"),
	py::arg("sigma")
	);
}