#pragma once

#include <cctype>
#include <initializer_list>
#include <istream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace naive_bayes::io {

class Json {
 public:
  using object_t = std::map<std::string, Json>;
  using array_t = std::vector<Json>;
  using string_t = std::string;
  using boolean_t = bool;
  using number_t = double;

  Json() : type_(Type::Null), data_(nullptr) {}
  Json(std::nullptr_t) : type_(Type::Null), data_(nullptr) {}
  Json(boolean_t value) : type_(Type::Boolean), data_(value) {}
  Json(number_t value) : type_(Type::Number), data_(value) {}
  Json(const string_t& value) : type_(Type::String), data_(value) {}
  Json(string_t&& value) : type_(Type::String), data_(std::move(value)) {}
  Json(const array_t& value) : type_(Type::Array), data_(value) {}
  Json(array_t&& value) : type_(Type::Array), data_(std::move(value)) {}
  Json(const object_t& value) : type_(Type::Object), data_(value) {}
  Json(object_t&& value) : type_(Type::Object), data_(std::move(value)) {}

  static Json object(std::initializer_list<std::pair<std::string, Json>> entries) {
    object_t obj;
    for (const auto& entry : entries) {
      obj.emplace(entry.first, entry.second);
    }
    return Json(std::move(obj));
  }

  static Json parse(std::istream& input) {
    std::string text;
    std::string chunk;
    while (std::getline(input, chunk)) {
      text.append(chunk);
      text.push_back('\n');
    }
    std::size_t pos = 0;
    skip_whitespace(text, pos);
    Json result = parse_value(text, pos);
    skip_whitespace(text, pos);
    if (pos != text.size()) {
      throw std::runtime_error("Unexpected trailing characters in JSON");
    }
    return result;
  }

  bool is_null() const { return type_ == Type::Null; }
  bool is_boolean() const { return type_ == Type::Boolean; }
  bool is_number() const { return type_ == Type::Number; }
  bool is_string() const { return type_ == Type::String; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_object() const { return type_ == Type::Object; }

  bool contains(const std::string& key) const {
    if (!is_object()) {
      return false;
    }
    const auto& obj = std::get<object_t>(data_);
    return obj.find(key) != obj.end();
  }

  const Json& at(const std::string& key) const {
    if (!is_object()) {
      throw std::runtime_error("JSON value is not an object");
    }
    const auto& obj = std::get<object_t>(data_);
    auto it = obj.find(key);
    if (it == obj.end()) {
      throw std::runtime_error("Key not found in JSON object: " + key);
    }
    return it->second;
  }

  const Json& at(std::size_t index) const {
    if (!is_array()) {
      throw std::runtime_error("JSON value is not an array");
    }
    const auto& arr = std::get<array_t>(data_);
    if (index >= arr.size()) {
      throw std::runtime_error("Array index out of range");
    }
    return arr[index];
  }

  std::size_t size() const {
    if (is_array()) {
      return std::get<array_t>(data_).size();
    }
    if (is_object()) {
      return std::get<object_t>(data_).size();
    }
    return 0;
  }

  const array_t& as_array() const {
    if (!is_array()) {
      throw std::runtime_error("JSON value is not an array");
    }
    return std::get<array_t>(data_);
  }

  const object_t& as_object() const {
    if (!is_object()) {
      throw std::runtime_error("JSON value is not an object");
    }
    return std::get<object_t>(data_);
  }

  template <typename T>
  T get() const {
    if constexpr (std::is_same_v<T, string_t>) {
      if (!is_string()) {
        throw std::runtime_error("JSON value is not a string");
      }
      return std::get<string_t>(data_);
    } else if constexpr (std::is_same_v<T, double>) {
      if (!is_number()) {
        throw std::runtime_error("JSON value is not a number");
      }
      return std::get<number_t>(data_);
    } else if constexpr (std::is_same_v<T, bool>) {
      if (!is_boolean()) {
        throw std::runtime_error("JSON value is not a boolean");
      }
      return std::get<boolean_t>(data_);
    } else if constexpr (std::is_same_v<T, std::size_t>) {
      if (!is_number()) {
        throw std::runtime_error("JSON value is not numeric");
      }
      return static_cast<std::size_t>(std::get<number_t>(data_));
    } else {
      static_assert(sizeof(T) == 0, "Unsupported Json::get<T>() type");
    }
  }

  template <typename T>
  T value(const std::string& key, const T& default_value) const {
    if (!is_object()) {
      return default_value;
    }
    const auto& obj = std::get<object_t>(data_);
    auto it = obj.find(key);
    if (it == obj.end()) {
      return default_value;
    }
    return it->second.template get<T>();
  }

 private:
  enum class Type { Null, Boolean, Number, String, Array, Object };

  Type type_;
  std::variant<std::nullptr_t, boolean_t, number_t, string_t, array_t, object_t> data_;

  static void skip_whitespace(const std::string& text, std::size_t& pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
      ++pos;
    }
  }

  static Json parse_value(const std::string& text, std::size_t& pos) {
    if (pos >= text.size()) {
      throw std::runtime_error("Unexpected end of JSON input");
    }
    char ch = text[pos];
    if (ch == '{') {
      return parse_object(text, pos);
    }
    if (ch == '[') {
      return parse_array(text, pos);
    }
    if (ch == '"') {
      return parse_string(text, pos);
    }
    if (ch == 't') {
      return parse_literal(text, pos, "true", Json(true));
    }
    if (ch == 'f') {
      return parse_literal(text, pos, "false", Json(false));
    }
    if (ch == 'n') {
      return parse_literal(text, pos, "null", Json(nullptr));
    }
    if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) {
      return parse_number(text, pos);
    }
    throw std::runtime_error("Invalid character in JSON input");
  }

  static Json parse_object(const std::string& text, std::size_t& pos) {
    object_t obj;
    ++pos;  // skip '{'
    skip_whitespace(text, pos);
    if (pos < text.size() && text[pos] == '}') {
      ++pos;
      return Json(obj);
    }
    while (pos < text.size()) {
      if (text[pos] != '"') {
        throw std::runtime_error("Expected string key in JSON object");
      }
      Json key = parse_string(text, pos);
      skip_whitespace(text, pos);
      if (pos >= text.size() || text[pos] != ':') {
        throw std::runtime_error("Expected ':' in JSON object");
      }
      ++pos;
      skip_whitespace(text, pos);
      Json value = parse_value(text, pos);
      obj.emplace(std::get<string_t>(key.data_), std::move(value));
      skip_whitespace(text, pos);
      if (pos >= text.size()) {
        throw std::runtime_error("Unexpected end of JSON object");
      }
      if (text[pos] == '}') {
        ++pos;
        break;
      }
      if (text[pos] != ',') {
        throw std::runtime_error("Expected ',' in JSON object");
      }
      ++pos;
      skip_whitespace(text, pos);
    }
    return Json(std::move(obj));
  }

  static Json parse_array(const std::string& text, std::size_t& pos) {
    array_t arr;
    ++pos;  // skip '['
    skip_whitespace(text, pos);
    if (pos < text.size() && text[pos] == ']') {
      ++pos;
      return Json(arr);
    }
    while (pos < text.size()) {
      Json value = parse_value(text, pos);
      arr.push_back(std::move(value));
      skip_whitespace(text, pos);
      if (pos >= text.size()) {
        throw std::runtime_error("Unexpected end of JSON array");
      }
      if (text[pos] == ']') {
        ++pos;
        break;
      }
      if (text[pos] != ',') {
        throw std::runtime_error("Expected ',' in JSON array");
      }
      ++pos;
      skip_whitespace(text, pos);
    }
    return Json(std::move(arr));
  }

  static Json parse_string(const std::string& text, std::size_t& pos) {
    std::string result;
    ++pos;  // skip opening quote
    while (pos < text.size()) {
      char ch = text[pos++];
      if (ch == '"') {
        return Json(result);
      }
      if (ch == '\\') {
        if (pos >= text.size()) {
          throw std::runtime_error("Invalid escape sequence in JSON string");
        }
        char esc = text[pos++];
        switch (esc) {
          case '"': result.push_back('"'); break;
          case '\\': result.push_back('\\'); break;
          case '/': result.push_back('/'); break;
          case 'b': result.push_back('\b'); break;
          case 'f': result.push_back('\f'); break;
          case 'n': result.push_back('\n'); break;
          case 'r': result.push_back('\r'); break;
          case 't': result.push_back('\t'); break;
          default:
            throw std::runtime_error("Unsupported escape sequence in JSON string");
        }
      } else {
        result.push_back(ch);
      }
    }
    throw std::runtime_error("Unterminated JSON string");
  }

  static Json parse_number(const std::string& text, std::size_t& pos) {
    std::size_t start = pos;
    if (text[pos] == '-') {
      ++pos;
    }
    if (pos >= text.size()) {
      throw std::runtime_error("Incomplete JSON number");
    }
    if (text[pos] == '0') {
      ++pos;
    } else if (std::isdigit(static_cast<unsigned char>(text[pos]))) {
      while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
        ++pos;
      }
    } else {
      throw std::runtime_error("Invalid JSON number");
    }
    if (pos < text.size() && text[pos] == '.') {
      ++pos;
      if (pos >= text.size() || !std::isdigit(static_cast<unsigned char>(text[pos]))) {
        throw std::runtime_error("Invalid JSON number");
      }
      while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
        ++pos;
      }
    }
    if (pos < text.size() && (text[pos] == 'e' || text[pos] == 'E')) {
      ++pos;
      if (pos < text.size() && (text[pos] == '+' || text[pos] == '-')) {
        ++pos;
      }
      if (pos >= text.size() || !std::isdigit(static_cast<unsigned char>(text[pos]))) {
        throw std::runtime_error("Invalid JSON number");
      }
      while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
        ++pos;
      }
    }

    double value = std::stod(text.substr(start, pos - start));
    return Json(value);
  }

  static Json parse_literal(const std::string& text, std::size_t& pos,
                            const std::string& literal, const Json& value) {
    if (text.compare(pos, literal.size(), literal) != 0) {
      throw std::runtime_error("Invalid literal in JSON input");
    }
    pos += literal.size();
    return value;
  }
};

}  // namespace naive_bayes::io
