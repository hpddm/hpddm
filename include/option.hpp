/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2015-07-21

   Copyright (C) 2015      Eidgenössische Technische Hochschule Zürich
                 2016-     Centre National de la Recherche Scientifique

   HPDDM is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published
   by the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   HPDDM is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with HPDDM.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _HPDDM_OPTION_
#define _HPDDM_OPTION_

#define HPDDM_PREFIX "hpddm_"
#define HPDDM_CONCAT(NAME) "" HPDDM_PREFIX #NAME ""

#include <stdlib.h>
#include <cstring>
#include <stdexcept>
#include <stack>
#ifndef HPDDM_NO_REGEX
#include <regex>
#endif
#include "singleton.hpp"

namespace HPDDM {
/* Class: Option
 *  A class to handle internal options of HPDDM or custom options as defined by the user in its application. */
class Option : private Singleton {
    private:
        /* Variable: opt
         *  Unordered map that stores the internal options of HPDDM. */
        std::unordered_map<std::string, double>  _opt;
        /* Variable: app
         *  Pointer to an unordered map that may store custom options as defined by the user in its application. */
        std::unordered_map<std::string, double>* _app;
        std::string                           _prefix;
        static bool hasEnding(const std::string& str, const std::string& ending) {
            return str.length() >= ending.length() ? str.compare(str.length() - ending.length(), ending.length(), ending) == 0 : false;
        }
    public:
        template<int N>
        explicit Option(Singleton::construct_key<N>);
        ~Option() {
            std::unordered_map<std::string, double>::const_iterator show = _opt.find("verbosity");
            if(show != _opt.cend()) {
                std::function<void(const std::unordered_map<std::string, double>&, const std::string&)> generate = [&](const std::unordered_map<std::string, double>& map, const std::string& header) {
                    std::vector<std::string> v;
                    v.reserve(map.size() + 3);
                    v.emplace_back(" ┌");
                    v.emplace_back(" │ " + header + " option" + std::string(map.size() > 1 ? "s" : "") + " used:");
                    size_t max = v.back().size();
                    for(const auto& x : map) {
                        double intpart;
                        if(x.second < -10000000 && x.first[-x.second - 10000000] == '#')
                            v.emplace_back(" │  " + x.first.substr(0, -x.second - 10000000) + ": " + x.first.substr(-x.second - 10000000 + 1));
                        else if(x.second < 1000 && std::modf(x.second, &intpart) == 0.0)
                            v.emplace_back(" │  " + x.first + ": " + to_string(static_cast<int>(x.second)));
                        else {
                            std::stringstream ss;
                            ss << std::scientific << std::setprecision(1) << x.second;
                            v.emplace_back(" │  " + x.first + ": " + ss.str());
                        }
                        max = std::max(max, v.back().size());
                    }
                    v.emplace_back(" └");
                    std::sort(v.begin() + 2, v.end() - 1, [](const std::string& a, const std::string& b) {
                        std::string::const_iterator p[2] { std::find_if(a.cbegin(), a.cend(), ::isdigit), std::find_if(b.cbegin(), b.cend(), ::isdigit) };
                        if(p[0] != a.cend() && p[1] != b.cend()) {
                            std::iterator_traits<std::string::const_iterator>::difference_type v[2] { std::distance(a.cbegin(), p[0]), std::distance(b.cbegin(), p[1]) };
                            if(a.substr(0, v[0]) == b.substr(0, v[1]))
                                return sto<int>(a.substr(v[0], a.size())) < sto<int>(b.substr(v[1], b.size()));
                            else
                                return a.substr(0, v[0]) < b.substr(0, v[1]);
                        }
                        else
                            return a < b;
                    });
                    output(v, max);
                };
                if(show->second > 1) {
                    if(_app)
                        generate(*_app, "Application-specific");
                    generate(_opt, "HPDDM");
                }
                if(show->second > 0 && _opt.find("version") != _opt.cend())
                    version();
            }
            delete _app;
        }
        static void output(const std::vector<std::string>& list, const size_t width) {
            std::cout << list.front() << std::setfill('-') << std::setw(width + 1) << std::right << "┐" << std::endl;
            for(std::vector<std::string>::const_iterator it = list.begin() + 1; it != list.end() - 1; ++it)
                std::cout << std::left << std::setfill(' ') << std::setw(width + 2) << *it << "│" << std::endl;
            std::cout << list.back() << std::setfill('-') << std::setw(width + 1) << std::right << "┘" << std::endl;
            std::cout << std::setfill(' ');
        }
        void version() const;
        /* Function: get
         *  Returns a shared pointer to <Option::opt>. */
        template<int N = 0>
        static std::shared_ptr<Option> get() {
            return Singleton::get<Option, N>();
        }
        /* Function: app
         *  Returns a constant reference of <Option::app>. */
        std::unordered_map<std::string, double>& app() const { return *_app; }
        bool set(const std::string& key) const { return _opt.find(_prefix + key) != _opt.cend(); }
        /* Function: remove
         *
         *  Removes a key from the unordered map <Option::opt>.
         *
         * Parameter:
         *    key            - Key to remove from <Option::opt>. */
        void remove(const std::string& key) {
            std::unordered_map<std::string, double>::const_iterator it = _opt.find(_prefix + key);
            if(it != _opt.cend())
                _opt.erase(it);
        }
        /* Function: val
         *  Returns the value of the key given as an argument, or use a default value if the key is not in <Option::opt>. */
        template<class T = double>
        T val(const std::string& key, T d = std::numeric_limits<T>::lowest()) const {
            std::unordered_map<std::string, double>::const_iterator it = _opt.find(_prefix + key);
            if(it == _opt.cend())
                return d;
            else
                return static_cast<T>(it->second);
        }
        const double& operator[](const std::string& key) const {
            try {
                return _opt.at(_prefix + key);
            }
            catch(const std::out_of_range& oor) {
                std::cerr << "out_of_range error: " << oor.what() << " (key: " << _prefix + key << ")" << std::endl;
                return _opt.cbegin()->second;
            }
        }
        double& operator[](const std::string& key) { return _opt[_prefix + key]; }
        struct Arg {
            static bool positive(const std::string& opt, const std::string& s, bool verbose) {
                if(!s.empty()) {
                    char* endptr = nullptr;
                    int val = strtol(s.c_str(), &endptr, 10);
                    if(endptr != s.c_str() && *endptr == 0 && !std::isnan(float(val)) && val > 0)
                        return true;
                }
                if(verbose)
                    std::cerr << "'" << opt << "' requires a positive integer argument ('" << s << "' was supplied)" << std::endl;
                return false;
            }
            static bool integer(const std::string& opt, const std::string& s, bool verbose) {
                if(!s.empty()) {
                    char* endptr = nullptr;
                    int val = strtol(s.c_str(), &endptr, 10);
                    if(endptr != s.c_str() && *endptr == 0 && !std::isnan(float(val)))
                        return true;
                }
                if(verbose)
                    std::cerr << "'" << opt << "' requires an integer argument ('" << s << "' was supplied)" << std::endl;
                return false;
            }
            static bool numeric(const std::string& opt, const std::string& s, bool verbose) {
                if(!s.empty()) {
                    char* endptr = nullptr;
                    double val = strtod(s.c_str(), &endptr);
                    if(endptr != s.c_str() && *endptr == 0 && !std::isnan(val))
                        return true;
                }
                if(verbose)
                    std::cerr << "'" << opt << "' requires a numeric argument ('" << s << "' was supplied)" << std::endl;
                return false;
            }
            static bool argument(const std::string& opt, const std::string& s, bool verbose) {
                if(!s.empty() && s[0] != '-')
                    return true;
                if(verbose)
                    std::cerr << "'" << opt << "' requires a valid argument ('" << s << "' was supplied)" << std::endl;
                return false;
            }
            static bool anything(const std::string&, const std::string&, bool) { return true; }
        };
        /* Function: prefix
         *
         *  Looks for a key in <Option::opt> or <Option::app> that starts with the prefix given as an argument.
         *
         * Parameter:
         *    pre            - Prefix to look for. */
        std::string prefix(const std::string& pre, const bool internal = false) const {
            if(!internal && !_app)
                return std::string();
            std::unordered_map<std::string, double>::const_iterator pIt[2];
            if(internal) {
                pIt[0] = _opt.cbegin();
                pIt[1] = _opt.cend();
            }
            else {
                pIt[0] = _app->cbegin();
                pIt[1] = _app->cend();
            }
            const std::string prefix = _prefix + pre;
            std::unordered_map<std::string, double>::const_iterator it = std::find_if(pIt[0], pIt[1], [&](const std::pair<std::string, double>& p) { bool match = std::mismatch(prefix.cbegin(), prefix.cend(), p.first.cbegin()).first == prefix.cend(); if(match && p.first.size() > prefix.size() + 1) { match = (p.first[prefix.size()] == '#'); } return match; });
            if(it != pIt[1])
                return it->first.substr(prefix.size() + 1);
            else
                return std::string();
        }
        /* Function: any_of
         *
         *  Returns true if the value of a given key in <Option::opt> takes its value in the std::initializer_list given as an argument.
         *
         * Parameters:
         *    key            - Key to look for.
         *    list           - List of values to search for. */
        template<class T>
        bool any_of(const std::string& key, std::initializer_list<T> list) const {
            std::unordered_map<std::string, double>::const_iterator it = _opt.find(key);
            return (it != _opt.cend() && std::any_of(list.begin(), list.end(), [&](const T& t) { return t == static_cast<T>(it->second); }));
        }
        template<class T, typename std::enable_if<std::is_same<T, char>::value || std::is_same<T, const char>::value>::type* = nullptr, class Container = std::initializer_list<std::tuple<std::string, std::string, std::function<bool(const std::string&, const std::string&, bool)>>>>
        int parse(int argc, T** argv, bool display = true, const Container& reg = { }) {
            std::vector<std::string> args(argv, argv + argc);
            return parse(args, display, reg);
        }
        template<class C, class Container = std::initializer_list<std::tuple<std::string, std::string, std::function<bool(const std::string&, const std::string&, bool)>>>, typename std::enable_if<!std::is_same<Container, bool>::value && !std::is_same<C, int>::value && !std::is_same<C, std::ifstream>::value>::type* = nullptr>
        int parse(C& arg, bool display = true, const Container& reg = { }) {
            std::vector<std::string> args;
            std::stringstream ss(arg);
            std::string item;
            while (std::getline(ss, item, ' '))
                args.push_back(item);
            return parse(args, display, reg);
        }
        int parse(std::ifstream& cfg, bool display = true) {
            if(!cfg.good()) {
                std::cout << "WARNING -- could not parse the supplied config file" << std::endl;
                return 1;
            }
            else {
                cfg.seekg(0, std::ios::end);
                size_t size = cfg.tellg();
                std::string buffer(size, ' ');
                cfg.seekg(0);
                cfg.read(&buffer[0], size);
                std::vector<std::string> s;
                s.reserve(size);
                std::stringstream ss(buffer);
                std::string item;
                while(std::getline(ss, item, '\n')) {
                    size = std::min(item.find("//"), item.find("#"));
                    if(size > 1 + std::string(HPDDM_PREFIX).size()) {
                        item = item.substr(0, size);
                        std::stringstream ws(item);
                        while(ws >> item) {
                            if(item.find("config_file") == std::string::npos)
                                s.emplace_back(item);
                        }
                    }
                }
                return parse<true>(s, display);
            }
        }
        template<bool = false, bool = false, class Container = std::initializer_list<std::tuple<std::string, std::string, std::function<bool(const std::string&, const std::string&, bool)>>>>
        int parse(std::vector<std::string>&, bool display = true, const Container& reg = { }, std::string prefix = "");
        template<bool internal, bool exact = false, class T>
        bool insert(const T& option, std::string str, const std::string& arg, const std::string& prefix = "") {
            static_assert(internal || !exact, "Wrong call");
            std::string::size_type n = str.find("=");
            bool sep = true;
            std::string val;
            if(n != std::string::npos) {
                sep = false;
                val = str.substr(n + 1);
                str = str.substr(0, n);
            }
            typename T::const_iterator it = std::find_if(option.begin(), option.end(), [&](typename T::const_reference tuple) {
                if(std::get<0>(tuple).empty())
                    return false;
                if(hasEnding(std::get<0>(tuple), str))
                    return exact ? std::get<0>(tuple).compare(str) == 0 : true;
                else {
                    std::string::size_type del = std::get<0>(tuple).find_first_of("=");
                    if(del != std::string::npos && del > 0 && std::get<0>(tuple)[del - 1] == '(')
                        --del;
                    if(hasEnding(str, std::get<0>(tuple).substr(0, del)))
                        return exact ? str.compare(std::get<0>(tuple).substr(0, del)) == 0 : true;
                    else {
#ifndef HPDDM_NO_REGEX
                        std::regex words_regex(std::get<0>(tuple).substr(0, del) + "$");
                        return std::sregex_iterator(str.cbegin(), str.cend(), words_regex) != std::sregex_iterator();
#else
                        return false;
#endif
                    }
                }
            });
            if(it != option.end()) {
                std::unordered_map<std::string, double>& map = (internal ? _opt : *_app);
                const bool boolean = (std::get<0>(*it).size() > 6 && std::get<0>(*it).substr(std::get<0>(*it).size() - 6) == "=(0|1)");
                const bool optional = std::get<0>(*it).find("(=") != std::string::npos;
                const std::string key = (exact ? prefix : "") + str;
                std::string empty;
                if(!std::get<2>(*it)(str, empty, false) || optional) {
                    bool success = true;
                    if(sep) {
                        if(arg.empty()) {
                            if(!optional && !boolean)
                                std::cout << "'" << str << "'" << " requires an argument" << std::endl;
                            else
                                map[key] = 1;
                            success = false;
                        }
                        else if(optional) {
                            if(Arg::numeric(str, arg, false)) {
                                map[key] = sto<double>(arg);
                                return true;
                            }
                            else
                                map[key] = 1;
                            success = false;
                        }
                        else if(std::get<2>(*it)(str, arg, !boolean)) {
                            val = arg;
                            std::string::size_type reg = std::get<0>(*it).find_first_of("=");
                            if(reg != std::string::npos && std::get<0>(*it).at(reg + 1) != '<')
                                empty = std::get<0>(*it).substr(reg + 1);
                        }
                        else {
                            if(boolean)
                                map[key] = 1;
                            success = false;
                        }
                    }
                    else if(optional) {
                        if(Arg::numeric(str, val, false))
                            map[key] = sto<double>(val);
                        else
                            map[key] = 1;
                        success = false;
                    }
                    else if(std::get<2>(*it)(str, val, true)) {
                        std::string::size_type reg = std::get<0>(*it).find_first_of("=");
                        if(reg != std::string::npos && std::get<0>(*it).at(reg + 1) != '<')
                            empty = std::get<0>(*it).substr(reg + 1);
                    }
                    else
                        success = false;
                    if(!empty.empty()) {
#ifndef HPDDM_NO_REGEX
                        std::regex words_regex(empty + "$", std::regex_constants::icase);
                        auto words_begin = std::sregex_iterator(val.cbegin(), val.cend(), words_regex);
                        if(std::distance(words_begin, std::sregex_iterator()) == 1) {
#else
                        std::string::size_type found = empty.find(val);
                        if(found != std::string::npos) {
#endif
                            char* endptr = nullptr;
                            double number = strtod(val.c_str(), &endptr);
                            if(endptr != empty.c_str() && *endptr == 0 && !std::isnan(number))
                                map[key] = number;
                            else {
#ifndef HPDDM_NO_REGEX
                                std::string::size_type found = empty.find(val);
                                if(found != std::string::npos)
#endif
                                    map[key] = std::count(empty.cbegin(), empty.cbegin() + found, '|');
#ifndef HPDDM_NO_REGEX
                                else
                                    std::cout << "WARNING -- something is wrong with this regular expression" << std::endl;
#endif
                            }
                            if(sep)
                                return true;
                        }
                        else {
                            if(boolean && (val.compare("true") == 0 || val.compare("yes") == 0))
                                map[key] = 1;
                            else if(boolean && (val.compare("false") == 0 || val.compare("no") == 0))
                                map[key];
                            else
                                std::cerr << "'" << val << "' doesn't match the regular expression '" << empty << "' for option '" << str << "'" << std::endl;
                        }
                    }
                    else if(success) {
#if __cpp_rtti || defined(__GXX_RTTI) || defined(__INTEL_RTTI__) || defined(_CPPRTTI)
                        auto target = std::get<2>(*it).template target<bool (*)(const std::string&, const std::string&, bool)>();
                        if(!target || *target != Arg::argument)
                            map[key] = sto<double>(val);
                        else {
                            for(const auto& x : map)
                                if(x.first.find(str + "#") == 0) {
                                    map.erase(x.first);
                                    break;
                                }
                            map[key + "#" + val] = -static_cast<int>(key.size()) - 10000000;
                        }
                        if(sep)
                            return true;
#else
                        try {
                            map[key] = sto<double>(val);
                            if(sep)
                                return true;
                        }
                        catch(const std::invalid_argument& ia) {
                            std::cerr << "invalid_argument error: " << ia.what() << " (key: " << str << ", value: " << val << ")" << std::endl;
                        }
#endif
                    }
                }
                else
                    map[key] = 1;
            }
            else if(internal && !exact)
                std::cout << "WARNING -- '-hpddm_" << str << "' is not a registered HPDDM option" << std::endl;
            return false;
        }
        void setPrefix(const std::string& pre) {
            _prefix = pre;
        }
        std::string getPrefix() const {
            return _prefix;
        }
};

class OptionsPrefix {
    protected:
        char* _prefix;
    public:
        OptionsPrefix() : _prefix() { };
        ~OptionsPrefix() {
            delete [] _prefix;
            _prefix = nullptr;
        }
        void setPrefix(const char* prefix) {
            if(_prefix)
                delete [] _prefix;
            _prefix = new char[std::strlen(prefix) + 1];
            std::strcpy(_prefix, prefix);
        }
        void setPrefix(const std::string& prefix) {
            if(prefix.size())
                setPrefix(prefix.c_str());
        }
        std::string prefix() const {
            return std::string(!_prefix ? "" : _prefix);
        }
        std::string prefix(const std::string& opt) const {
            return !_prefix ? opt : std::string(_prefix) + opt;
        }
};
} // HPDDM
#endif // _HPDDM_OPTION_
