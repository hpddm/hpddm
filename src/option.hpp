/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@inf.ethz.ch>
        Date: 2015-07-21

   Copyright (C) 2015      Eidgenössische Technische Hochschule Zürich

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

#ifndef _OPTION_
#define _OPTION_

#define HPDDM_PREFIX "hpddm_"
#define HPDDM_CONCAT(NAME) "" HPDDM_PREFIX #NAME ""

#if !defined(HPDDM_NO_REGEX)
#include <regex>
#endif

namespace HPDDM {
/* Class: Option
 *  A class to handle internal options of HPDDM or custom options as defined by the user in its application. */
class Option {
    private:
        /* Variable: opt
         *  Unordered map that stores the internal options of HPDDM. */
        std::unordered_map<std::string, double>  _opt;
        /* Variable: app
         *  Pointer to an unordered map that may store custom options as defined by the user in its application. */
        std::unordered_map<std::string, double>* _app;
        class construct_key { };
    public:
        Option(construct_key);
        Option(const Option&) = delete;
        ~Option() {
            if(_opt.find("verbosity") != _opt.cend()) {
                std::function<void(const std::unordered_map<std::string, double>&, const std::string&)> output = [](const std::unordered_map<std::string, double>& map, const std::string& header) {
                    std::vector<std::string> output;
                    output.reserve(map.size() + 3);
                    output.emplace_back(" ┌");
                    output.emplace_back(" │ " + header + " option" + std::string(map.size() > 1 ? "s" : "") + " used:");
                    size_t max = output.back().size();
                    for(const auto& x : map) {
                        double intpart;
                        if(x.second < -10000000 && x.first[-x.second - 10000000] == '_')
                            output.emplace_back(" │  " + x.first.substr(0, -x.second - 10000000) + ": " + x.first.substr(-x.second - 10000000 + 1));
                        else if(x.second < 1000 && std::modf(x.second, &intpart) == 0.0)
                            output.emplace_back(" │  " + x.first + ": " + to_string(static_cast<int>(x.second)));
                        else {
                            std::stringstream ss;
                            ss << std::scientific << std::setprecision(1) << x.second;
                            output.emplace_back(" │  " + x.first + ": " + ss.str());
                        }
                        max = std::max(max, output.back().size());
                    }
                    output.emplace_back(" └");
                    std::sort(output.begin() + 2, output.end() - 1, [](const std::string& a, const std::string& b) {
                        std::pair<std::string::const_iterator, std::string::const_iterator> p(std::find_if(a.cbegin(), a.cend(), ::isdigit), std::find_if(b.cbegin(), b.cend(), ::isdigit));
                        if(p.first != a.cend() && p.second != b.cend()) {
                            std::pair<size_t, size_t> v(std::distance(a.begin(), p.first), std::distance(b.begin(), p.second));
                            if(a.substr(0, v.first) == b.substr(0, v.second))
                                return sto<int>(a.substr(v.first, a.size())) < sto<int>(b.substr(v.second, b.size()));
                            else
                                return a.substr(0, v.first) < b.substr(0, v.second);
                        }
                        else
                            return a < b;
                    });
                    std::cout << output.front() << std::setfill('-') << std::setw(max + 1) << std::right << "┐" << std::endl;
                    for(std::vector<std::string>::const_iterator it = output.begin() + 1; it != output.end() - 1; ++it)
                        std::cout << std::left << std::setfill(' ') << std::setw(max + 2) << *it << "│" << std::endl;
                    std::cout << output.back() << std::setfill('-') << std::setw(max + 1) << std::right << "┘" << std::endl;
                    std::cout << std::setfill(' ');
                };
                if(_app)
                    output(*_app, "Application-specific");
                output(_opt, "HPDDM");
            }
            delete _app;
        }
        /* Function: get
         *  Returns a shared pointer to <Option::opt>. */
        static std::shared_ptr<Option> get() {
            static std::shared_ptr<Option> instance = std::make_shared<Option>(construct_key());
            return instance;
        }
        /* Function: app
         *  Returns a constant reference of <Option::app>. */
        std::unordered_map<std::string, double>& app() const { return *_app; }
        bool set(const std::string& key) const { return _opt.find(key) != _opt.cend(); }
        /* Function: remove
         *
         *  Removes a key from the unordered map <Option::opt>.
         *
         * Parameter:
         *    key            - Key to remove for <Option::opt>. */
        void remove(const std::string& key) { _opt.erase(key); }
        /* Function: val
         *
         *  Returns the value of the key given as an argument, or use a default value if the key is not in <Option::opt>.
         *
         * Parameter:
         *    key            - Key to remove for <Option::opt>. */
        double val(const std::string& key, double d = std::numeric_limits<double>::lowest()) const {
            std::unordered_map<std::string, double>::const_iterator it = _opt.find(key);
            if(it == _opt.cend())
                return d;
            else
                return it->second;
        }
        const double& operator[](const std::string& key) const { return _opt.at(key); }
        double& operator[](const std::string& key) { return _opt[key]; }
        struct Arg {
            static bool integer(const std::string& opt, const std::string& s, bool verbose) {
                char* endptr = nullptr;
                auto inter = std::bind(strtol, std::placeholders::_1, std::placeholders::_2, 10);
                if(!s.empty()) {
                    inter(s.c_str(), &endptr);
                    if(endptr != s.c_str() && *endptr == 0)
                        return true;
                }
                if(verbose)
                    std::cerr << "'" << opt << "' requires an integer argument ('" << s << "' was supplied)" << std::endl;
                return false;
            }
            static bool numeric(const std::string& opt, const std::string& s, bool verbose) {
                char* endptr = nullptr;
                if(!s.empty()) {
                    strtod(s.c_str(), &endptr);
                    if(endptr != s.c_str() && *endptr == 0)
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
        std::string prefix(const std::string& pre, bool internal = false) const {
            if(!internal && _app == nullptr)
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
            std::unordered_map<std::string, double>::const_iterator it = std::find_if(pIt[0], pIt[1], [&](const std::pair<std::string, double>& p) { return std::mismatch(pre.begin(), pre.end(), p.first.begin()).first == pre.end(); });
            if(it != pIt[1] && it->first.size() > pre.size() + 1)
                return it->first.substr(pre.size() + 1);
            else
                return std::string();
        }
        template<class T, typename std::enable_if<std::is_same<T, char>::value || std::is_same<T, const char>::value>::type* = nullptr>
        int parse(int argc, T** argv, bool display = true, std::initializer_list<std::tuple<std::string, std::string, std::function<bool(const std::string&, const std::string&, bool)>>> reg = { }) {
            std::vector<std::string> args(argv + 1, argv + argc);
            return parse(args, display, reg);
        }
        int parse(std::string& arg, bool display = true, std::initializer_list<std::tuple<std::string, std::string, std::function<bool(const std::string&, const std::string&, bool)>>> reg = { }) {
            std::vector<std::string> args(1, arg);
            return parse(args, display, reg);
        }
        int parse(std::vector<std::string>&, bool display = true, std::initializer_list<std::tuple<std::string, std::string, std::function<bool(const std::string&, const std::string&, bool)>>> reg = { });
        template<class T>
        void insert(std::unordered_map<std::string, double>& map, const T& option, std::string& str, const std::string& arg) {
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
                if(std::get<0>(tuple) == str)
                    return true;
                else {
                    std::string::size_type del = std::get<0>(tuple).find_first_of("=");
                    if(del != std::string::npos && del > 0 && std::get<0>(tuple)[del - 1] == '(')
                        --del;
                    if(std::get<0>(tuple).substr(0, del) == str)
                        return true;
                    else {
#if !defined(HPDDM_NO_REGEX)
                        std::regex words_regex("^" + std::get<0>(tuple).substr(0, del) + "$");
                        return std::sregex_iterator(str.cbegin(), str.cend(), words_regex) != std::sregex_iterator();
#else
                        return false;
#endif
                    }
                }
            });
            if(it != option.end()) {
                std::string empty;
                bool optional = std::get<0>(*it).find("(=") != std::string::npos;
                if(!std::get<2>(*it)(str, empty, false) || optional) {
                    bool success = true;
                    if(sep) {
                        if(arg.empty()) {
                            if(!optional)
                                std::cout << "'" << str << "'" << " requires an argument" << std::endl;
                            else
                                map[str];
                            success = false;
                        }
                        else if(optional) {
                            if(Arg::numeric(str, arg, false))
                                map[str] = sto<double>(arg);
                            else
                                map[str];
                            success = false;
                        }
                        else if(std::get<2>(*it)(str, arg, true)) {
                            val = arg;
                            std::string::size_type reg = std::get<0>(*it).find_first_of("=");
                            if(reg != std::string::npos && std::get<0>(*it).at(reg + 1) != '<')
                                empty = std::get<0>(*it).substr(reg + 1);
                        }
                        else
                            success = false;
                    }
                    else if(optional) {
                        if(Arg::numeric(str, val, false))
                            map[str] = sto<double>(val);
                        else
                            map[str];
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
#if !defined(HPDDM_NO_REGEX)
                        std::regex words_regex(val, std::regex_constants::icase);
                        auto words_begin = std::sregex_iterator(empty.cbegin(), empty.cend(), words_regex);
                        if(std::distance(words_begin, std::sregex_iterator()) == 1)
                            map[str] = std::count(empty.begin(), empty.begin() + words_begin->position(), '|');
#else
                        std::string::size_type found = empty.find(val);
                        if(found != std::string::npos)
                            map[str] = std::count(empty.begin(), empty.begin() + found, '|');
#endif
                        else
                            std::cerr << "'" << val << "' doesn't match the regular expression '" << empty << "' for option '" << str << "'" << std::endl;
                    }
                    else if(success) {
                        auto target = std::get<2>(*it).template target<bool (*)(const std::string&, const std::string&, bool)>();
                        if(!target || (*target != Arg::argument))
                            map[str] = sto<double>(val);
                        else
                            map[str + "_" + val] = -static_cast<int>(str.size()) - 10000000;
                    }
                }
                else
                    map[str];
            }
        }
};
} // HPDDM
#endif // _OPTION_
