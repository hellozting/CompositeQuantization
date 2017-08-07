#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>

using std::string;
using std::ifstream;
using std::map;
using std::cout;
using std::endl;

class CQParameters
{
public:
	/**
	* The constructor function.
	*/
	CQParameters();

	/**
	* The deconstructor function.
	*/
	~CQParameters();


	/**
	* This function tests whether the value of a parameter key exists.
	*  @param  key    A parameter.
	*/
	bool Exists(const string& key);

	/**
	* This function outputs parameter information when help is used in the config.txt.
	*/
	void WriteHelpInformation();

	/**
	* This function load the parameters from txt file.
	*  @param  parameter_file   The filename that stores all the parameters.
	*/
	void LoadFromFile(const string parameter_file);


	/**
	* This template function gets the value of parameter key.
	*  @param  key    The parameter.
	*  @param  val    The value of the parameter key.
	*/
	template<class T>
	void Get(const string& key, T& val)
	{
		map<string, string>::const_iterator p = parameter_set.find(key);
		if (p != parameter_set.end())
		{
			val = StringToValue<T>(p->second);
		}
		else
		{
			cout << "Error : can not find the parameter " + key << endl;
			throw std::logic_error("the parameter can't be found: " + key);
		}
	}

	/**
	* This template function returns the value of parameter key.
	*  @param  key    The parameter.
	*/
	template<class T>
	 const T Get(const string& key)
	{
		T val;
		map<string, string>::const_iterator p = parameter_set.find(key);
		if (p != parameter_set.end())
		{
			val = StringToValue<T>(p->second);
		}
		else
		{
			cout << "Error : can not find the parameter " + key << endl;
			throw std::logic_error("the parameter can't be found: " + key);
		}
		return val;
	}


	/**
	* This template function sets the value of parameter key.
	*  @param  key    The parameter.
	*  @param  val    The value of the parameter key.
	*/
	template<class T>
	void Set(const string& key, const T& val)
	{
		parameter_set[key] = ValueToString(val);
	}


	/**
	* This template function converts a string to a value and returns it.
	*  @param  str    The value stored in string.
	*/
	template<typename T>
	T StringToValue(const string& str);
	
	/**
	* string (The explicit specialization definition).
	*/
	template<> string StringToValue<string>(const string& str)
	{
		return string(str);
	}

	/**
	* int (The explicit specialization definition).
	*/
	template<> int StringToValue<int>(const string& str)
	{
		return atoi(str.c_str());
	}

	/**
	* float (The explicit specialization definition).
	*/
	template<> float StringToValue<float>(const string& str)
	{
		return float(atof(str.c_str()));
	}

	/**
	* double (The explicit specialization definition).
	*/
	template<> double StringToValue<double>(const string& str)
	{
		return atof(str.c_str());
	}


	/**
	* This template function converts a value to a string and returns it.
	*  @param  val    The value to be coverted.
	*/
	template<typename T>
	string ValueToString(const T& val)
	{
		return std::to_string(T);
	}

	/**
	* string (The explicit specialization definition).
	*/
	template<> string ValueToString<string>(const string& val)
	{
		return val;
	}

private:
	/**
	* The set of parameters.
	*/
	map<string, string> parameter_set;
};