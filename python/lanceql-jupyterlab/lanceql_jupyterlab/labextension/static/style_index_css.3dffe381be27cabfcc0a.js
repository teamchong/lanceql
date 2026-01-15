"use strict";
(self["webpackChunklanceql_jupyterlab"] = self["webpackChunklanceql_jupyterlab"] || []).push([["style_index_css"],{

/***/ "./node_modules/css-loader/dist/cjs.js!./style/index.css"
/*!***************************************************************!*\
  !*** ./node_modules/css-loader/dist/cjs.js!./style/index.css ***!
  \***************************************************************/
(module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "default": () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _node_modules_css_loader_dist_runtime_sourceMaps_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ../node_modules/css-loader/dist/runtime/sourceMaps.js */ "./node_modules/css-loader/dist/runtime/sourceMaps.js");
/* harmony import */ var _node_modules_css_loader_dist_runtime_sourceMaps_js__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_node_modules_css_loader_dist_runtime_sourceMaps_js__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _node_modules_css_loader_dist_runtime_api_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ../node_modules/css-loader/dist/runtime/api.js */ "./node_modules/css-loader/dist/runtime/api.js");
/* harmony import */ var _node_modules_css_loader_dist_runtime_api_js__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_node_modules_css_loader_dist_runtime_api_js__WEBPACK_IMPORTED_MODULE_1__);
// Imports


var ___CSS_LOADER_EXPORT___ = _node_modules_css_loader_dist_runtime_api_js__WEBPACK_IMPORTED_MODULE_1___default()((_node_modules_css_loader_dist_runtime_sourceMaps_js__WEBPACK_IMPORTED_MODULE_0___default()));
// Module
___CSS_LOADER_EXPORT___.push([module.id, `/* LanceQL Virtual Table Styles */

.lanceql-virtual-table {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-size: 13px;
  line-height: 1.4;
}

.lq-container {
  border: 1px solid var(--jp-border-color1, #e0e0e0);
  border-radius: 4px;
  overflow: hidden;
  background: var(--jp-layout-color0, #fff);
}

/* Header */
.lq-header {
  display: flex;
  background: var(--jp-layout-color2, #f5f5f5);
  font-weight: 600;
  border-bottom: 1px solid var(--jp-border-color1, #e0e0e0);
  position: sticky;
  top: 0;
  z-index: 10;
}

.lq-header .lq-cell {
  padding: 8px 12px;
  color: var(--jp-ui-font-color1, #333);
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.lq-col-name {
  font-weight: 600;
}

/* Type badges */
.lq-type-badge {
  display: inline-block;
  font-size: 10px;
  font-weight: 500;
  padding: 1px 6px;
  border-radius: 3px;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  width: fit-content;
}

.lq-type-vector {
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  color: white;
}

.lq-type-string {
  background: #10b981;
  color: white;
}

.lq-type-int {
  background: #3b82f6;
  color: white;
}

.lq-type-float {
  background: #f59e0b;
  color: white;
}

.lq-type-bool {
  background: #ec4899;
  color: white;
}

.lq-type-timestamp {
  background: #14b8a6;
  color: white;
}

.lq-type-list {
  background: #8b5cf6;
  color: white;
}

.lq-type-other,
.lq-type-unknown {
  background: #6b7280;
  color: white;
}

/* Scroll area */
.lq-scroll {
  height: 400px;
  overflow-y: auto;
  overflow-x: auto;
}

.lq-spacer {
  position: relative;
  min-width: 100%;
}

.lq-rows {
  position: relative;
}

/* Rows */
.lq-row {
  display: flex;
  border-bottom: 1px solid var(--jp-border-color3, #f0f0f0);
  position: absolute;
  width: 100%;
  box-sizing: border-box;
  background: var(--jp-layout-color0, #fff);
}

.lq-row:hover {
  background: var(--jp-layout-color1, #f9f9f9);
}

.lq-row.lq-loading {
  color: var(--jp-ui-font-color3, #999);
  font-style: italic;
}

/* Cells */
.lq-cell {
  flex: 1;
  padding: 6px 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 80px;
  max-width: 300px;
  color: var(--jp-ui-font-color1, #333);
  position: relative;
}

/* Image cells with hover preview */
.lq-cell.lq-img-cell {
  position: relative;
  overflow: visible;  /* Allow preview to show */
}

.lq-cell.lq-img-cell a {
  color: var(--jp-brand-color1, #0066cc);
  text-decoration: none;
  cursor: pointer;
}

.lq-cell.lq-img-cell a:hover {
  text-decoration: underline;
}

/* Image preview is now rendered via JS to document.body */

/* Status bar */
.lq-status {
  padding: 4px 12px;
  background: var(--jp-layout-color1, #fafafa);
  color: var(--jp-ui-font-color2, #666);
  font-size: 11px;
  border-top: 1px solid var(--jp-border-color2, #e0e0e0);
}

/* Dark theme support via JupyterLab CSS variables */
[data-jp-theme-light='false'] .lq-container {
  border-color: var(--jp-border-color1);
}

[data-jp-theme-light='false'] .lq-header {
  background: var(--jp-layout-color2);
  border-color: var(--jp-border-color1);
}

[data-jp-theme-light='false'] .lq-row {
  border-color: var(--jp-border-color3);
  background: var(--jp-layout-color0);
}

[data-jp-theme-light='false'] .lq-row:hover {
  background: var(--jp-layout-color1);
}

/* Vector cells with sparklines */
.lq-vector-cell {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
}

.lq-vector-cell:hover {
  background: var(--jp-layout-color1, #f0f0f0);
}

.lq-sparkline {
  flex-shrink: 0;
}

.lq-dim-label {
  font-size: 10px;
  color: var(--jp-ui-font-color2, #666);
  background: var(--jp-layout-color2, #f0f0f0);
  padding: 1px 4px;
  border-radius: 2px;
}

/* Vector stats popup */
.lq-vector-stats {
  min-width: 180px;
  font-size: 12px;
}

.lq-stats-header {
  font-weight: 600;
  padding-bottom: 8px;
  margin-bottom: 8px;
  border-bottom: 1px solid var(--jp-border-color2, #e0e0e0);
  color: var(--jp-ui-font-color1, #333);
}

.lq-stats-row {
  display: flex;
  justify-content: space-between;
  padding: 3px 0;
  color: var(--jp-ui-font-color2, #666);
}

.lq-stats-row span:first-child {
  font-weight: 500;
}

.lq-stats-row span:last-child {
  font-family: 'SF Mono', Menlo, monospace;
  color: var(--jp-ui-font-color1, #333);
}

/* Dark theme for vector elements */
[data-jp-theme-light='false'] .lq-dim-label {
  background: var(--jp-layout-color2);
  color: var(--jp-ui-font-color2);
}

[data-jp-theme-light='false'] .lq-sparkline path {
  stroke: #818cf8;
}

[data-jp-theme-light='false'] .lq-vector-stats {
  background: var(--jp-layout-color1);
}

[data-jp-theme-light='false'] .lq-stats-header {
  border-color: var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
}

[data-jp-theme-light='false'] .lq-stats-row span:last-child {
  color: var(--jp-ui-font-color1);
}
`, "",{"version":3,"sources":["webpack://./style/index.css"],"names":[],"mappings":"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","sourcesContent":["/* LanceQL Virtual Table Styles */\n\n.lanceql-virtual-table {\n  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;\n  font-size: 13px;\n  line-height: 1.4;\n}\n\n.lq-container {\n  border: 1px solid var(--jp-border-color1, #e0e0e0);\n  border-radius: 4px;\n  overflow: hidden;\n  background: var(--jp-layout-color0, #fff);\n}\n\n/* Header */\n.lq-header {\n  display: flex;\n  background: var(--jp-layout-color2, #f5f5f5);\n  font-weight: 600;\n  border-bottom: 1px solid var(--jp-border-color1, #e0e0e0);\n  position: sticky;\n  top: 0;\n  z-index: 10;\n}\n\n.lq-header .lq-cell {\n  padding: 8px 12px;\n  color: var(--jp-ui-font-color1, #333);\n  display: flex;\n  flex-direction: column;\n  gap: 4px;\n}\n\n.lq-col-name {\n  font-weight: 600;\n}\n\n/* Type badges */\n.lq-type-badge {\n  display: inline-block;\n  font-size: 10px;\n  font-weight: 500;\n  padding: 1px 6px;\n  border-radius: 3px;\n  text-transform: uppercase;\n  letter-spacing: 0.3px;\n  width: fit-content;\n}\n\n.lq-type-vector {\n  background: linear-gradient(135deg, #6366f1, #8b5cf6);\n  color: white;\n}\n\n.lq-type-string {\n  background: #10b981;\n  color: white;\n}\n\n.lq-type-int {\n  background: #3b82f6;\n  color: white;\n}\n\n.lq-type-float {\n  background: #f59e0b;\n  color: white;\n}\n\n.lq-type-bool {\n  background: #ec4899;\n  color: white;\n}\n\n.lq-type-timestamp {\n  background: #14b8a6;\n  color: white;\n}\n\n.lq-type-list {\n  background: #8b5cf6;\n  color: white;\n}\n\n.lq-type-other,\n.lq-type-unknown {\n  background: #6b7280;\n  color: white;\n}\n\n/* Scroll area */\n.lq-scroll {\n  height: 400px;\n  overflow-y: auto;\n  overflow-x: auto;\n}\n\n.lq-spacer {\n  position: relative;\n  min-width: 100%;\n}\n\n.lq-rows {\n  position: relative;\n}\n\n/* Rows */\n.lq-row {\n  display: flex;\n  border-bottom: 1px solid var(--jp-border-color3, #f0f0f0);\n  position: absolute;\n  width: 100%;\n  box-sizing: border-box;\n  background: var(--jp-layout-color0, #fff);\n}\n\n.lq-row:hover {\n  background: var(--jp-layout-color1, #f9f9f9);\n}\n\n.lq-row.lq-loading {\n  color: var(--jp-ui-font-color3, #999);\n  font-style: italic;\n}\n\n/* Cells */\n.lq-cell {\n  flex: 1;\n  padding: 6px 12px;\n  overflow: hidden;\n  text-overflow: ellipsis;\n  white-space: nowrap;\n  min-width: 80px;\n  max-width: 300px;\n  color: var(--jp-ui-font-color1, #333);\n  position: relative;\n}\n\n/* Image cells with hover preview */\n.lq-cell.lq-img-cell {\n  position: relative;\n  overflow: visible;  /* Allow preview to show */\n}\n\n.lq-cell.lq-img-cell a {\n  color: var(--jp-brand-color1, #0066cc);\n  text-decoration: none;\n  cursor: pointer;\n}\n\n.lq-cell.lq-img-cell a:hover {\n  text-decoration: underline;\n}\n\n/* Image preview is now rendered via JS to document.body */\n\n/* Status bar */\n.lq-status {\n  padding: 4px 12px;\n  background: var(--jp-layout-color1, #fafafa);\n  color: var(--jp-ui-font-color2, #666);\n  font-size: 11px;\n  border-top: 1px solid var(--jp-border-color2, #e0e0e0);\n}\n\n/* Dark theme support via JupyterLab CSS variables */\n[data-jp-theme-light='false'] .lq-container {\n  border-color: var(--jp-border-color1);\n}\n\n[data-jp-theme-light='false'] .lq-header {\n  background: var(--jp-layout-color2);\n  border-color: var(--jp-border-color1);\n}\n\n[data-jp-theme-light='false'] .lq-row {\n  border-color: var(--jp-border-color3);\n  background: var(--jp-layout-color0);\n}\n\n[data-jp-theme-light='false'] .lq-row:hover {\n  background: var(--jp-layout-color1);\n}\n\n/* Vector cells with sparklines */\n.lq-vector-cell {\n  display: flex;\n  align-items: center;\n  gap: 6px;\n  cursor: pointer;\n}\n\n.lq-vector-cell:hover {\n  background: var(--jp-layout-color1, #f0f0f0);\n}\n\n.lq-sparkline {\n  flex-shrink: 0;\n}\n\n.lq-dim-label {\n  font-size: 10px;\n  color: var(--jp-ui-font-color2, #666);\n  background: var(--jp-layout-color2, #f0f0f0);\n  padding: 1px 4px;\n  border-radius: 2px;\n}\n\n/* Vector stats popup */\n.lq-vector-stats {\n  min-width: 180px;\n  font-size: 12px;\n}\n\n.lq-stats-header {\n  font-weight: 600;\n  padding-bottom: 8px;\n  margin-bottom: 8px;\n  border-bottom: 1px solid var(--jp-border-color2, #e0e0e0);\n  color: var(--jp-ui-font-color1, #333);\n}\n\n.lq-stats-row {\n  display: flex;\n  justify-content: space-between;\n  padding: 3px 0;\n  color: var(--jp-ui-font-color2, #666);\n}\n\n.lq-stats-row span:first-child {\n  font-weight: 500;\n}\n\n.lq-stats-row span:last-child {\n  font-family: 'SF Mono', Menlo, monospace;\n  color: var(--jp-ui-font-color1, #333);\n}\n\n/* Dark theme for vector elements */\n[data-jp-theme-light='false'] .lq-dim-label {\n  background: var(--jp-layout-color2);\n  color: var(--jp-ui-font-color2);\n}\n\n[data-jp-theme-light='false'] .lq-sparkline path {\n  stroke: #818cf8;\n}\n\n[data-jp-theme-light='false'] .lq-vector-stats {\n  background: var(--jp-layout-color1);\n}\n\n[data-jp-theme-light='false'] .lq-stats-header {\n  border-color: var(--jp-border-color2);\n  color: var(--jp-ui-font-color1);\n}\n\n[data-jp-theme-light='false'] .lq-stats-row span:last-child {\n  color: var(--jp-ui-font-color1);\n}\n"],"sourceRoot":""}]);
// Exports
/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (___CSS_LOADER_EXPORT___);


/***/ },

/***/ "./node_modules/css-loader/dist/runtime/api.js"
/*!*****************************************************!*\
  !*** ./node_modules/css-loader/dist/runtime/api.js ***!
  \*****************************************************/
(module) {



/*
  MIT License http://www.opensource.org/licenses/mit-license.php
  Author Tobias Koppers @sokra
*/
module.exports = function (cssWithMappingToString) {
  var list = [];

  // return the list of modules as css string
  list.toString = function toString() {
    return this.map(function (item) {
      var content = "";
      var needLayer = typeof item[5] !== "undefined";
      if (item[4]) {
        content += "@supports (".concat(item[4], ") {");
      }
      if (item[2]) {
        content += "@media ".concat(item[2], " {");
      }
      if (needLayer) {
        content += "@layer".concat(item[5].length > 0 ? " ".concat(item[5]) : "", " {");
      }
      content += cssWithMappingToString(item);
      if (needLayer) {
        content += "}";
      }
      if (item[2]) {
        content += "}";
      }
      if (item[4]) {
        content += "}";
      }
      return content;
    }).join("");
  };

  // import a list of modules into the list
  list.i = function i(modules, media, dedupe, supports, layer) {
    if (typeof modules === "string") {
      modules = [[null, modules, undefined]];
    }
    var alreadyImportedModules = {};
    if (dedupe) {
      for (var k = 0; k < this.length; k++) {
        var id = this[k][0];
        if (id != null) {
          alreadyImportedModules[id] = true;
        }
      }
    }
    for (var _k = 0; _k < modules.length; _k++) {
      var item = [].concat(modules[_k]);
      if (dedupe && alreadyImportedModules[item[0]]) {
        continue;
      }
      if (typeof layer !== "undefined") {
        if (typeof item[5] === "undefined") {
          item[5] = layer;
        } else {
          item[1] = "@layer".concat(item[5].length > 0 ? " ".concat(item[5]) : "", " {").concat(item[1], "}");
          item[5] = layer;
        }
      }
      if (media) {
        if (!item[2]) {
          item[2] = media;
        } else {
          item[1] = "@media ".concat(item[2], " {").concat(item[1], "}");
          item[2] = media;
        }
      }
      if (supports) {
        if (!item[4]) {
          item[4] = "".concat(supports);
        } else {
          item[1] = "@supports (".concat(item[4], ") {").concat(item[1], "}");
          item[4] = supports;
        }
      }
      list.push(item);
    }
  };
  return list;
};

/***/ },

/***/ "./node_modules/css-loader/dist/runtime/sourceMaps.js"
/*!************************************************************!*\
  !*** ./node_modules/css-loader/dist/runtime/sourceMaps.js ***!
  \************************************************************/
(module) {



module.exports = function (item) {
  var content = item[1];
  var cssMapping = item[3];
  if (!cssMapping) {
    return content;
  }
  if (typeof btoa === "function") {
    var base64 = btoa(unescape(encodeURIComponent(JSON.stringify(cssMapping))));
    var data = "sourceMappingURL=data:application/json;charset=utf-8;base64,".concat(base64);
    var sourceMapping = "/*# ".concat(data, " */");
    return [content].concat([sourceMapping]).join("\n");
  }
  return [content].join("\n");
};

/***/ },

/***/ "./node_modules/style-loader/dist/runtime/injectStylesIntoStyleTag.js"
/*!****************************************************************************!*\
  !*** ./node_modules/style-loader/dist/runtime/injectStylesIntoStyleTag.js ***!
  \****************************************************************************/
(module) {



var stylesInDOM = [];
function getIndexByIdentifier(identifier) {
  var result = -1;
  for (var i = 0; i < stylesInDOM.length; i++) {
    if (stylesInDOM[i].identifier === identifier) {
      result = i;
      break;
    }
  }
  return result;
}
function modulesToDom(list, options) {
  var idCountMap = {};
  var identifiers = [];
  for (var i = 0; i < list.length; i++) {
    var item = list[i];
    var id = options.base ? item[0] + options.base : item[0];
    var count = idCountMap[id] || 0;
    var identifier = "".concat(id, " ").concat(count);
    idCountMap[id] = count + 1;
    var indexByIdentifier = getIndexByIdentifier(identifier);
    var obj = {
      css: item[1],
      media: item[2],
      sourceMap: item[3],
      supports: item[4],
      layer: item[5]
    };
    if (indexByIdentifier !== -1) {
      stylesInDOM[indexByIdentifier].references++;
      stylesInDOM[indexByIdentifier].updater(obj);
    } else {
      var updater = addElementStyle(obj, options);
      options.byIndex = i;
      stylesInDOM.splice(i, 0, {
        identifier: identifier,
        updater: updater,
        references: 1
      });
    }
    identifiers.push(identifier);
  }
  return identifiers;
}
function addElementStyle(obj, options) {
  var api = options.domAPI(options);
  api.update(obj);
  var updater = function updater(newObj) {
    if (newObj) {
      if (newObj.css === obj.css && newObj.media === obj.media && newObj.sourceMap === obj.sourceMap && newObj.supports === obj.supports && newObj.layer === obj.layer) {
        return;
      }
      api.update(obj = newObj);
    } else {
      api.remove();
    }
  };
  return updater;
}
module.exports = function (list, options) {
  options = options || {};
  list = list || [];
  var lastIdentifiers = modulesToDom(list, options);
  return function update(newList) {
    newList = newList || [];
    for (var i = 0; i < lastIdentifiers.length; i++) {
      var identifier = lastIdentifiers[i];
      var index = getIndexByIdentifier(identifier);
      stylesInDOM[index].references--;
    }
    var newLastIdentifiers = modulesToDom(newList, options);
    for (var _i = 0; _i < lastIdentifiers.length; _i++) {
      var _identifier = lastIdentifiers[_i];
      var _index = getIndexByIdentifier(_identifier);
      if (stylesInDOM[_index].references === 0) {
        stylesInDOM[_index].updater();
        stylesInDOM.splice(_index, 1);
      }
    }
    lastIdentifiers = newLastIdentifiers;
  };
};

/***/ },

/***/ "./node_modules/style-loader/dist/runtime/insertBySelector.js"
/*!********************************************************************!*\
  !*** ./node_modules/style-loader/dist/runtime/insertBySelector.js ***!
  \********************************************************************/
(module) {



var memo = {};

/* istanbul ignore next  */
function getTarget(target) {
  if (typeof memo[target] === "undefined") {
    var styleTarget = document.querySelector(target);

    // Special case to return head of iframe instead of iframe itself
    if (window.HTMLIFrameElement && styleTarget instanceof window.HTMLIFrameElement) {
      try {
        // This will throw an exception if access to iframe is blocked
        // due to cross-origin restrictions
        styleTarget = styleTarget.contentDocument.head;
      } catch (e) {
        // istanbul ignore next
        styleTarget = null;
      }
    }
    memo[target] = styleTarget;
  }
  return memo[target];
}

/* istanbul ignore next  */
function insertBySelector(insert, style) {
  var target = getTarget(insert);
  if (!target) {
    throw new Error("Couldn't find a style target. This probably means that the value for the 'insert' parameter is invalid.");
  }
  target.appendChild(style);
}
module.exports = insertBySelector;

/***/ },

/***/ "./node_modules/style-loader/dist/runtime/insertStyleElement.js"
/*!**********************************************************************!*\
  !*** ./node_modules/style-loader/dist/runtime/insertStyleElement.js ***!
  \**********************************************************************/
(module) {



/* istanbul ignore next  */
function insertStyleElement(options) {
  var element = document.createElement("style");
  options.setAttributes(element, options.attributes);
  options.insert(element, options.options);
  return element;
}
module.exports = insertStyleElement;

/***/ },

/***/ "./node_modules/style-loader/dist/runtime/setAttributesWithoutAttributes.js"
/*!**********************************************************************************!*\
  !*** ./node_modules/style-loader/dist/runtime/setAttributesWithoutAttributes.js ***!
  \**********************************************************************************/
(module, __unused_webpack_exports, __webpack_require__) {



/* istanbul ignore next  */
function setAttributesWithoutAttributes(styleElement) {
  var nonce =  true ? __webpack_require__.nc : 0;
  if (nonce) {
    styleElement.setAttribute("nonce", nonce);
  }
}
module.exports = setAttributesWithoutAttributes;

/***/ },

/***/ "./node_modules/style-loader/dist/runtime/styleDomAPI.js"
/*!***************************************************************!*\
  !*** ./node_modules/style-loader/dist/runtime/styleDomAPI.js ***!
  \***************************************************************/
(module) {



/* istanbul ignore next  */
function apply(styleElement, options, obj) {
  var css = "";
  if (obj.supports) {
    css += "@supports (".concat(obj.supports, ") {");
  }
  if (obj.media) {
    css += "@media ".concat(obj.media, " {");
  }
  var needLayer = typeof obj.layer !== "undefined";
  if (needLayer) {
    css += "@layer".concat(obj.layer.length > 0 ? " ".concat(obj.layer) : "", " {");
  }
  css += obj.css;
  if (needLayer) {
    css += "}";
  }
  if (obj.media) {
    css += "}";
  }
  if (obj.supports) {
    css += "}";
  }
  var sourceMap = obj.sourceMap;
  if (sourceMap && typeof btoa !== "undefined") {
    css += "\n/*# sourceMappingURL=data:application/json;base64,".concat(btoa(unescape(encodeURIComponent(JSON.stringify(sourceMap)))), " */");
  }

  // For old IE
  /* istanbul ignore if  */
  options.styleTagTransform(css, styleElement, options.options);
}
function removeStyleElement(styleElement) {
  // istanbul ignore if
  if (styleElement.parentNode === null) {
    return false;
  }
  styleElement.parentNode.removeChild(styleElement);
}

/* istanbul ignore next  */
function domAPI(options) {
  if (typeof document === "undefined") {
    return {
      update: function update() {},
      remove: function remove() {}
    };
  }
  var styleElement = options.insertStyleElement(options);
  return {
    update: function update(obj) {
      apply(styleElement, options, obj);
    },
    remove: function remove() {
      removeStyleElement(styleElement);
    }
  };
}
module.exports = domAPI;

/***/ },

/***/ "./node_modules/style-loader/dist/runtime/styleTagTransform.js"
/*!*********************************************************************!*\
  !*** ./node_modules/style-loader/dist/runtime/styleTagTransform.js ***!
  \*********************************************************************/
(module) {



/* istanbul ignore next  */
function styleTagTransform(css, styleElement) {
  if (styleElement.styleSheet) {
    styleElement.styleSheet.cssText = css;
  } else {
    while (styleElement.firstChild) {
      styleElement.removeChild(styleElement.firstChild);
    }
    styleElement.appendChild(document.createTextNode(css));
  }
}
module.exports = styleTagTransform;

/***/ },

/***/ "./style/index.css"
/*!*************************!*\
  !*** ./style/index.css ***!
  \*************************/
(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "default": () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _node_modules_style_loader_dist_runtime_injectStylesIntoStyleTag_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! !../node_modules/style-loader/dist/runtime/injectStylesIntoStyleTag.js */ "./node_modules/style-loader/dist/runtime/injectStylesIntoStyleTag.js");
/* harmony import */ var _node_modules_style_loader_dist_runtime_injectStylesIntoStyleTag_js__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_node_modules_style_loader_dist_runtime_injectStylesIntoStyleTag_js__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _node_modules_style_loader_dist_runtime_styleDomAPI_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! !../node_modules/style-loader/dist/runtime/styleDomAPI.js */ "./node_modules/style-loader/dist/runtime/styleDomAPI.js");
/* harmony import */ var _node_modules_style_loader_dist_runtime_styleDomAPI_js__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_node_modules_style_loader_dist_runtime_styleDomAPI_js__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _node_modules_style_loader_dist_runtime_insertBySelector_js__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! !../node_modules/style-loader/dist/runtime/insertBySelector.js */ "./node_modules/style-loader/dist/runtime/insertBySelector.js");
/* harmony import */ var _node_modules_style_loader_dist_runtime_insertBySelector_js__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(_node_modules_style_loader_dist_runtime_insertBySelector_js__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var _node_modules_style_loader_dist_runtime_setAttributesWithoutAttributes_js__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! !../node_modules/style-loader/dist/runtime/setAttributesWithoutAttributes.js */ "./node_modules/style-loader/dist/runtime/setAttributesWithoutAttributes.js");
/* harmony import */ var _node_modules_style_loader_dist_runtime_setAttributesWithoutAttributes_js__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(_node_modules_style_loader_dist_runtime_setAttributesWithoutAttributes_js__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var _node_modules_style_loader_dist_runtime_insertStyleElement_js__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! !../node_modules/style-loader/dist/runtime/insertStyleElement.js */ "./node_modules/style-loader/dist/runtime/insertStyleElement.js");
/* harmony import */ var _node_modules_style_loader_dist_runtime_insertStyleElement_js__WEBPACK_IMPORTED_MODULE_4___default = /*#__PURE__*/__webpack_require__.n(_node_modules_style_loader_dist_runtime_insertStyleElement_js__WEBPACK_IMPORTED_MODULE_4__);
/* harmony import */ var _node_modules_style_loader_dist_runtime_styleTagTransform_js__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! !../node_modules/style-loader/dist/runtime/styleTagTransform.js */ "./node_modules/style-loader/dist/runtime/styleTagTransform.js");
/* harmony import */ var _node_modules_style_loader_dist_runtime_styleTagTransform_js__WEBPACK_IMPORTED_MODULE_5___default = /*#__PURE__*/__webpack_require__.n(_node_modules_style_loader_dist_runtime_styleTagTransform_js__WEBPACK_IMPORTED_MODULE_5__);
/* harmony import */ var _node_modules_css_loader_dist_cjs_js_index_css__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! !!../node_modules/css-loader/dist/cjs.js!./index.css */ "./node_modules/css-loader/dist/cjs.js!./style/index.css");

      
      
      
      
      
      
      
      
      

var options = {};

options.styleTagTransform = (_node_modules_style_loader_dist_runtime_styleTagTransform_js__WEBPACK_IMPORTED_MODULE_5___default());
options.setAttributes = (_node_modules_style_loader_dist_runtime_setAttributesWithoutAttributes_js__WEBPACK_IMPORTED_MODULE_3___default());

      options.insert = _node_modules_style_loader_dist_runtime_insertBySelector_js__WEBPACK_IMPORTED_MODULE_2___default().bind(null, "head");
    
options.domAPI = (_node_modules_style_loader_dist_runtime_styleDomAPI_js__WEBPACK_IMPORTED_MODULE_1___default());
options.insertStyleElement = (_node_modules_style_loader_dist_runtime_insertStyleElement_js__WEBPACK_IMPORTED_MODULE_4___default());

var update = _node_modules_style_loader_dist_runtime_injectStylesIntoStyleTag_js__WEBPACK_IMPORTED_MODULE_0___default()(_node_modules_css_loader_dist_cjs_js_index_css__WEBPACK_IMPORTED_MODULE_6__["default"], options);




       /* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (_node_modules_css_loader_dist_cjs_js_index_css__WEBPACK_IMPORTED_MODULE_6__["default"] && _node_modules_css_loader_dist_cjs_js_index_css__WEBPACK_IMPORTED_MODULE_6__["default"].locals ? _node_modules_css_loader_dist_cjs_js_index_css__WEBPACK_IMPORTED_MODULE_6__["default"].locals : undefined);


/***/ }

}]);
//# sourceMappingURL=style_index_css.3dffe381be27cabfcc0a.js.map