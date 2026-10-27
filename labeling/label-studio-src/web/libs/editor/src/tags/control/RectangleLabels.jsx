import { observer } from "mobx-react";
import { types } from "mobx-state-tree";

import LabelMixin from "../../mixins/LabelMixin";
import Registry from "../../core/Registry";
import SelectedModelMixin from "../../mixins/SelectedModel";
import Types from "../../core/Types";
import { HtxLabels, LabelsModel } from "./Labels/Labels";
import { RectangleModel } from "./Rectangle";
import { guidGenerator } from "../../core/Helpers";
import ControlBase from "./Base";

/**
 * The `RectangleLabels` tag creates labeled rectangles.
 *
 * Dynamic class support:
 *
 * - No application class is hard-coded.
 * - Classes can be created at runtime from arbitrary user input.
 * - The newly created LabelModel is selected through Label Studio's
 *   native label selection mechanism.
 * - A box drawn afterwards therefore receives the real class
 *   immediately.
 */

const Validation = types.model({
  controlledTags: Types.unionTag(["Image"]),
});

const ModelAttrs = types.model("RectangleLabelsModel", {
  pid: types.optional(types.string, guidGenerator),
  type: "rectanglelabels",
  children: Types.unionArray(["label", "header", "view", "hypertext"]),
});

const DynamicClassActions = types
  .model({})
  .actions((self) => ({
    applyDynamicClass(rawValue) {
      const className = String(rawValue ?? "")
        .replace(/\s+/g, " ")
        .trim();

      if (!className) {
        return false;
      }

      /*
       * Find an existing native LabelModel first.
       */
      let label = self.findLabel(className);

      /*
       * Arbitrary user class does not exist yet:
       * create a real LabelModel in RectangleLabels.
       */
      if (!label) {
        /*
         * An empty <RectangleLabels> can be parsed with
         * children === null. Initialise it only when the
         * annotator creates the first real class.
         */
        if (!self.children) {
          self.children = [];
        }

        self.children.push({
          type: "label",
          id: guidGenerator(),
          value: className,
          children: [],
        });

        label = self.findLabel(className);
      }

      if (!label) {
        console.error(
          "[DynamicClass] Unable to create label:",
          className
        );
        return false;
      }

      /*
       * If this label is already active we keep it active.
       *
       * Otherwise use Label Studio's native toggleSelected().
       *
       * No selected region:
       *   -> class becomes active drawing class.
       *
       * Selected region:
       *   -> native Label Studio relabels that region.
       *
       * choice="single":
       *   -> old class is replaced by this class.
       */
      if (!label.selected) {
        label.toggleSelected();
      }

      /*
       * Observable children.push() should already trigger rerender,
       * but this hook helps versions that expose it.
       */
      self.needsUpdate?.();

      return true;
    },

    getDynamicClassValues() {
      return self.tiedChildren
        .map((label) => label.value)
        .filter((value) => Boolean(value));
    },
  }));

const Composition = types.compose(
  ControlBase,
  LabelsModel,
  ModelAttrs,
  DynamicClassActions,
  RectangleModel,
  Validation,
  LabelMixin,
  SelectedModelMixin.props({ _child: "LabelModel" }),
);

const RectangleLabelsModel = types.compose(
  "RectangleLabelsModel",
  Composition
);

const HtxRectangleLabels = observer(({ item }) => {
  /*
   * Tiny bridge for the custom toolbar included by Django.
   *
   * className is ALWAYS supplied by the annotator.
   * There are no hard-coded domain classes here.
   */
  if (typeof window !== "undefined") {
    window.__LS_DYNAMIC_CLASS_BRIDGE__ = {
      applyClass(className) {
        return item.applyDynamicClass(className);
      },

      getClasses() {
        return item.getDynamicClassValues();
      },

      getSelectedClasses() {
        return item.selectedValues();
      },
    };
  }

  return <HtxLabels item={item} />;
});

Registry.addTag(
  "rectanglelabels",
  RectangleLabelsModel,
  HtxRectangleLabels
);

export {
  HtxRectangleLabels,
  RectangleLabelsModel,
};
