import { getRoot } from "mobx-state-tree";
import { AnnotationPreview } from "../Common/AnnotationPreview/AnnotationPreview";

const imgDefaultProps = { crossOrigin: "anonymous", loading: "lazy", decoding: "async" };

const toThumbnailUrl = (src) => {
  if (typeof src !== "string" || !src.includes("/data/upload/") || /[?&]thumb=/.test(src)) {
    return src;
  }
  const sep = src.includes("?") ? "&" : "?";
  return `${src}${sep}thumb=1&w=240`;
};

export const ImageCell = (column) => {
  const {
    original,
    value,
    column: { alias },
  } = column;
  const root = getRoot(original);

  const renderImagePreview = original.total_annotations === 0 || !root.showPreviews;
  const imgSrc = Array.isArray(value) ? value[0] : value;

  if (!imgSrc) return null;

  const thumbSrc = toThumbnailUrl(imgSrc);

  return renderImagePreview ? (
    <img
      {...imgDefaultProps}
      key={thumbSrc}
      src={thumbSrc}
      alt="Data"
      style={{
        maxHeight: "100%",
        maxWidth: "100px",
        objectFit: "contain",
        borderRadius: 3,
      }}
    />
  ) : (
    <AnnotationPreview
      task={original}
      annotation={original.annotations[0]}
      config={getRoot(original).SDK}
      name={alias}
      variant="120x120"
      fallbackImage={thumbSrc}
      style={{
        maxHeight: "100%",
        maxWidth: "100px",
        objectFit: "contain",
        borderRadius: 3,
      }}
    />
  );
};
