// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

using pair_t = std::pair<std::string, std::string>;

struct argsparse
{
public:
// options : option list (option name, brief explanation on the option)
// defaults: default value list (option name, default value)
  argsparse(const int argc, const char* const argv[],
    const std::vector<pair_t> & options,
    const std::vector<pair_t> & defaults = std::vector<pair_t>()):
    options_(options.size()), args_(options.size())
  {
    bool raiseError = false;
    std::vector<std::string> strArgv(argc-1);
    std::vector<bool> areOptionsChecked(options.size(), false);
    // separating options from the cmd line
    for (int i=1; i<argc; ++i)
      strArgv[i-1] = std::string(argv[i]);
    for (int i=0; i<options_.size(); ++i)
      options_[i] = options[i].first;
    for (const auto & str : strArgv)
    {
      if (str.compare("--help") == 0)
      {
        std::cout << " # option list" << std::endl;
        for (const auto & option : options)
        {
          std::cout << std::setw(8) << (option.first) << " : " << option.second;
          for (const auto defv : defaults)
          {
            if (defv.first.compare(option.first) == 0)
              std::cout << " (default : " << defv.second << ")";
          }
          std::cout << std::endl;
        }
        std::cout << std::endl << " (hint) " << argv[0]
		              << " -option1=value1 -option2=value2 ..." << std::endl;
        exit(1);
      }
    }
    // searching an argument
    for (const auto & str : strArgv)
    {
      for (int i=0; i<options_.size(); ++i)
      {
        if (options_[i].compare(str.substr(1, options_[i].length())) == 0)
        {
          if (str.length() <= 2 + options_[i].length())
          {
            std::cerr << "# error(in-1) ---> Put the option correctly! : " << options_[i] << std::endl;
            raiseError = true;
            break;
          }
          if (areOptionsChecked[i])
          {
            std::cerr << "# error(in-2) ---> The doubly occupied option is found! : " << options_[i] << ": " << str << std::endl;
            raiseError = true;
            break;
          }
          if (str.substr(1+options_[i].length(),1).compare("=") != 0)
          {
            std::cerr << "# error(in-3) ---> The symbol '=' must be between the option and the argument. : " << str << std::endl;
            raiseError = true;
            break;
          }
          const std::string arg = str.substr(2+options_[i].length());
          args_[i] = arg;
          areOptionsChecked[i] = true;
          break;
        }
      }
    }
    // inserting a default value
    for (int i=0; i<options_.size(); ++i)
    {
      if (!areOptionsChecked[i])
      {
        for (const auto & defv : defaults)
        {
          if (options_[i].compare(defv.first) == 0)
          {
            args_[i] = defv.second;
            areOptionsChecked[i] = true;
            break;
          }
        }
      }
    }

    for (int i=0; i<options_.size(); ++i)
    {
      if (!areOptionsChecked[i])
      {
        std::cerr << "# error(in) ---> The following option is missing. : " << options_[i] << std::endl;
        raiseError = true;
      }
    }

    if (raiseError)
    {
      std::cerr << std::endl << " (hint) Type the command to the cmd line as follows: "
                << argv[0] << " --help" << std::endl;
      exit(1);
    }
  }

  // calling an option
  template <typename T = std::string>
  T find(const std::string callOption) const
  {
    bool raiseError = true;
    std::string returnValue;
    for (int i=0; i<options_.size(); ++i)
    {
      if (callOption.compare(options_[i]) == 0)
      {
        returnValue = args_[i];
        raiseError = false;
        break;
      }
    }
    if (raiseError)
    {
      std::cerr << "# error(out) ---> Threre is no option for your calling. : " << callOption << std::endl;
      exit(1);
    }
    std::stringstream os;
    T value;
    os << returnValue;
    os >> value;
    if (os.fail())
    {
      std::cerr << "# error has occured in the lexical cast: " << returnValue << std::endl;
      exit(1);
    }
    return value;
  }

  // printing a current status
  template <typename FstreamType>
  void print(FstreamType & fstream) const
  {
    fstream << "#===== updated arguments =====" << std::endl;
    for (int i=0; i<options_.size(); ++i)
    {
      fstream << "# " << std::setw(8)
              << options_[i] << " : "
              << args_[i] << std::endl;
    }
    fstream << "#=============================" << std::endl;
  }

private:
  std::vector<std::string> options_, args_;
};
