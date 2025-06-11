#include <iostream>
#include <cstdio>
#include <vector>
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <numpy/ndarrayobject.h>
#include "boost/tuple/tuple.hpp"
#include "boost/python/object.hpp"
#include <../include/API.h>
//#include <../include/connected_components.h>

//#include <easy3d/PTCloud.hpp>
#include <easy3d/kdtree.h>
#include <easy3d/point_cloud.h>
#include <easy3d/point_cloud_io.h>

namespace bpn = boost::numpy;
namespace bp =  boost::python;

typedef boost::tuple< std::vector< std::vector<uint32_t> >, std::vector<uint32_t> > Custom_tuple;

struct VecToArray
{//converts a vector<uint32_t> to a numpy array
    static PyObject * convert(const std::vector<uint32_t> & vec)
    {
        npy_intp dims = vec.size();
        PyObject * obj = PyArray_SimpleNew(1, &dims, NPY_UINT32);
        void * arr_data = PyArray_DATA((PyArrayObject*)obj);
        memcpy(arr_data, &vec[0], dims * sizeof(uint32_t));
        return obj;
    }
};

struct VecToArray_float
{//converts a vector<uint32_t> to a numpy array
    static PyObject * convert(const std::vector<float> & vec)
    {
        npy_intp dims = vec.size();
        PyObject * obj = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
        void * arr_data = PyArray_DATA((PyArrayObject*)obj);
        memcpy(arr_data, &vec[0], dims * sizeof(float));
        return obj;
    }
};


template<class T>
struct VecvecToList
{//converts a vector< vector<T> > to a list
        static PyObject* convert(const std::vector< std::vector<T> > & vecvec)
    {
        boost::python::list* pylistlist = new boost::python::list();
        for(size_t i = 0; i < vecvec.size(); i++)
        {
            boost::python::list* pylist = new boost::python::list();
            for(size_t j = 0; j < vecvec[i].size(); j++)
            {
                pylist->append(vecvec[i][j]);
            }
            pylistlist->append((pylist, pylist[0]));
        }
        return pylistlist->ptr();
    }
};

struct to_py_tuple
{//converts output to a python tuple
    static PyObject* convert(const Custom_tuple& c_tuple){
        bp::list values;
        //add all c_tuple items to "values" list

        PyObject * vecvec_pyo = VecvecToList<uint32_t>::convert(c_tuple.get<0>());
        PyObject * vec_pyo = VecToArray::convert(c_tuple.get<1>());

        values.append(bp::handle<>(bp::borrowed(vecvec_pyo)));
        values.append(bp::handle<>(bp::borrowed(vec_pyo)));

        return bp::incref( bp::tuple( values ).ptr() );
    }
};

PyObject * cutpursuit(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target,const bpn::ndarray & edge_weight,
                      float lambda, const int cutoff, const int spatial, float weight_decay)
{//read data and run the L0-cut pursuit partition algorithm
    srand(0);

    const uint32_t n_ver = bp::len(obs);
    const uint32_t n_edg = bp::len(source);
    const uint32_t n_obs = bp::len(obs[0]);
    const float * obs_data = reinterpret_cast<float*>(obs.get_data());
    const uint32_t * source_data = reinterpret_cast<uint32_t*>(source.get_data());
    const uint32_t * target_data = reinterpret_cast<uint32_t*>(target.get_data());
    const float * edge_weight_data = reinterpret_cast<float*>(edge_weight.get_data());
    std::vector<float> solution(n_ver *n_obs);
    //float solution [n_ver * n_obs];
    std::vector<float> node_weight(n_ver, 1.0f);
    std::vector<uint32_t> in_component(n_ver,0);
    std::vector< std::vector<uint32_t> > components(1,std::vector<uint32_t>(1,0.f));
    if (spatial == 0)
    {
        CP::cut_pursuit<float>(n_ver, n_edg, n_obs, obs_data, source_data, target_data, edge_weight_data, &node_weight[0]
                 , solution.data(), in_component, components, lambda, (uint32_t)cutoff,  1.f, 4.f, weight_decay, 0.f);
    }
    else
    {
        CP::cut_pursuit<float>(n_ver, n_edg, n_obs, obs_data, source_data, target_data, edge_weight_data, &node_weight[0]
                 , solution.data(), in_component, components, lambda, (uint32_t)cutoff,  2.f, 4.f, weight_decay, 0.f);
    }
    return to_py_tuple::convert(Custom_tuple(components, in_component));
}

PyObject * cutpursuit2(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target,const bpn::ndarray & edge_weight,
                       const bpn::ndarray & node_weight, float lambda)
{//read data and run the L0-cut pursuit partition algorithm
    srand(0);
std::cout << std::endl;
    const uint32_t n_ver = bp::len(obs);
    const uint32_t n_edg = bp::len(source);
    const uint32_t n_obs = bp::len(obs[0]);
    const float * obs_data = reinterpret_cast<float*>(obs.get_data());
    const uint32_t * source_data = reinterpret_cast<uint32_t*>(source.get_data());
    const uint32_t * target_data = reinterpret_cast<uint32_t*>(target.get_data());
    const float * edge_weight_data = reinterpret_cast<float*>(edge_weight.get_data());
    const float * node_weight_data = reinterpret_cast<float*>(node_weight.get_data());
    std::vector<float> solution(n_ver *n_obs);
    //float solution [n_ver * n_obs];
    //std::vector<float> node_weight(n_ver, 1.0f);
    std::vector<uint32_t> in_component(n_ver,0);
    std::vector< std::vector<uint32_t> > components(1,std::vector<uint32_t>(1,0.f));
    CP::cut_pursuit<float>(n_ver, n_edg, n_obs, obs_data, source_data, target_data, edge_weight_data, node_weight_data
                 , solution.data(), in_component, components, lambda, (uint32_t)0,  2.f, 4.f, 1.f, 1.f);
    return to_py_tuple::convert(Custom_tuple(components, in_component));
}

//read point cloud data and parsing to python array
PyObject *  pointlcoud_parsing
(
	const bp::str read_data_path_py,
	bpn::ndarray & py_x,
	bpn::ndarray & py_y,
	bpn::ndarray & py_z,
	bpn::ndarray & py_r,
	bpn::ndarray & py_g,
	bpn::ndarray & py_b,
	bpn::ndarray & py_labels
)
{
	//read *.ply
	std::ostringstream pcl_str_ostemp;
	std::string read_data_path = bp::extract<std::string>(read_data_path_py);
	pcl_str_ostemp << read_data_path;
	std::string pcl_str_temp = pcl_str_ostemp.str().data();
	char * pclPath_temp = (char *)pcl_str_temp.data();
	easy3d::PointCloud* pcl_in = easy3d::PointCloudIO::load(pclPath_temp);

	//parsing from ply to std::vector
	auto point_coord = pcl_in->get_vertex_property<easy3d::vec3>("v:point");
	auto point_color = pcl_in->get_vertex_property<easy3d::vec3>("v:color");
	auto point_segid = pcl_in->get_vertex_property<int>("v:point_segment_id");
	auto point_label = pcl_in->get_vertex_property<int>("v:point_groundtruth_label");

	std::vector<float> c_x, c_y, c_z;
	std::vector<uint8_t> c_r, c_g, c_b;
	std::vector<uint32_t> c_labels;
	std::vector<uint32_t> in_component;
	for (auto ptx : pcl_in->vertices())
	{
		c_x.push_back(point_coord[ptx].x);
		c_y.push_back(point_coord[ptx].y);
		c_z.push_back(point_coord[ptx].z);

		c_r.push_back(uint8_t(point_color[ptx].x * 255.0f));
		c_g.push_back(uint8_t(point_color[ptx].y * 255.0f));
		c_b.push_back(uint8_t(point_color[ptx].z * 255.0f));

		c_labels.push_back(point_label[ptx]);
		in_component.push_back(point_segid[ptx]);
	}
	sort(in_component.begin(), in_component.end());
	in_component.erase(std::unique(in_component.begin(), in_component.end()), in_component.end());

	std::vector< std::vector<uint32_t> > components(in_component.size(), std::vector<uint32_t>());
	for (auto ptx : pcl_in->vertices())
		components[point_segid[ptx]].push_back(ptx.idx());

	//from std::vector to numpy array
	int sizeOfs_xyz_rgb = c_x.size();
	float *originals_x = c_x.data(), *originals_y = c_y.data(), *originals_z = c_z.data();
	uint8_t *originals_r = c_r.data(), *originals_g = c_g.data(), *originals_b = c_b.data();
	uint32_t *originals_labels = c_labels.data();

	bpn::dtype s_xyz_type = bpn::dtype::get_builtin<float>();
	bpn::dtype s_rgb_type = bpn::dtype::get_builtin<uint8_t>();
	bpn::dtype s_labels_type = bpn::dtype::get_builtin<uint32_t>();

	py_x = bpn::from_data
	(
		originals_x, 
		s_xyz_type,
		bp::make_tuple(1, sizeOfs_xyz_rgb),//rows, cols
		bp::make_tuple(sizeof(float) * 0, sizeof(float) * 1),
		bp::object()
	);

	py_y = bpn::from_data
	(
		originals_y,
		s_xyz_type,
		bp::make_tuple(1, sizeOfs_xyz_rgb),//rows, cols
		bp::make_tuple(sizeof(float) * 0, sizeof(float) * 1),
		bp::object()
	);

	py_z = bpn::from_data
	(
		originals_z,
		s_xyz_type,
		bp::make_tuple(1, sizeOfs_xyz_rgb),//rows, cols
		bp::make_tuple(sizeof(float) * 0, sizeof(float) * 1),
		bp::object()
	);

	py_r = bpn::from_data
	(
		originals_r,
		s_rgb_type,
		bp::make_tuple(1, sizeOfs_xyz_rgb),//rows, cols
		bp::make_tuple(sizeof(uint8_t) * 0, sizeof(uint8_t) * 1),
		bp::object()
	);

	py_g = bpn::from_data
	(
		originals_g,
		s_rgb_type,
		bp::make_tuple(1, sizeOfs_xyz_rgb),//rows, cols
		bp::make_tuple(sizeof(uint8_t) * 0, sizeof(uint8_t) * 1),
		bp::object()
	);

	py_b = bpn::from_data
	(
		originals_b,
		s_rgb_type,
		bp::make_tuple(1, sizeOfs_xyz_rgb),//rows, cols
		bp::make_tuple(sizeof(uint8_t) * 0, sizeof(uint8_t) * 1),
		bp::object()
	);

	py_labels = bpn::from_data
	(
		originals_labels,
		s_labels_type,
		bp::make_tuple(1, sizeOfs_xyz_rgb),//rows, cols
		bp::make_tuple(sizeof(int) * 0, sizeof(int) * 1),
		bp::object()
	);

	return to_py_tuple::convert(Custom_tuple(components, in_component));
}

BOOST_PYTHON_MODULE(libpp)
{
    _import_array();
    Py_Initialize();
    bpn::initialize();
    bp::to_python_converter< Custom_tuple, to_py_tuple>();
    
    def("cutpursuit", cutpursuit);
    def("cutpursuit", cutpursuit, (bp::args("cutoff")=0, bp::args("spatial")=0, bp::args("weight_decay")=1));
    def("cutpursuit2", cutpursuit2);

	def("pointlcoud_parsing", pointlcoud_parsing);
}

