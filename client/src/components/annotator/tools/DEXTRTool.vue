<script>
import paper from "paper";
import tool from "@/mixins/toolBar/tool";
import axios from "axios";

export default {
  name: "RITMTool",
  mixins: [tool],
  props: {
    scale: {
      type: Number,
      default: 1
    }
  },
  data() {
    return {
      icon: "fa-crosshairs",
      name: "RITM",
      cursor: "crosshair",
      settings: {
        padding: 50,
        threshold: 80
      },
      previous_points_count: 0,
      points: [],
      fg_points: [],
      bg_points: []
    };
  },
  methods: {
    createFGPoint(point) {
      let paperPoint = new paper.Path.Circle(point, 5);
      // paperPoint.fillColor = this.$parent.currentAnnotation.color;
      paperPoint.fillColor = "#50ff40";
      paperPoint.data.point = point;
      this.points.push(paperPoint);
      this.fg_points.push(paperPoint);
    },
    createbBGPoint(point) {
      let paperPoint = new paper.Path.Circle(point, 5);
      paperPoint.fillColor = "red";
      paperPoint.data.point = point;
      this.points.push(paperPoint);
      this.bg_points.push(paperPoint);
    },
    onMouseDown(event) {
      if (Key.isDown('a')) {
        this.createbBGPoint(event.point)
      } else {
        this.createFGPoint(event.point);
      }
    },
    clearPoints() {
      this.points.forEach(point => point.remove());
      this.fg_points.forEach(point => point.remove());
      this.bg_points.forEach(point => point.remove());
      this.points = [];
      this.fg_points = [];
      this.bg_points = [];
      this.previous_points_count = 0;
    },
    onKeyDown(event) {
      if (Key.isDown('f')) {
        console.error('Clear points data');
        this.clearPoints();
      }
    }
  },
  computed: {
    isDisabled() {
      this.clearPoints();
      return this.$parent.current.annotation == -1;
    }
  },
  watch: {
    points(newPoints) {
      if (newPoints.length > 0 && this.fg_points.length > 0 && newPoints.length !== this.previous_points_count) {
        this.previous_points_count = newPoints.length;

        // let bg_points = this.bg_points;
        // let fg_points = this.fg_points;
        // this.points = [];
        let currentAnnotation = this.$parent.currentAnnotation;
        let pointsList = [];
        let width = this.$parent.image.raster.width / 2;
        let height = this.$parent.image.raster.height / 2;

        this.bg_points.forEach(point => {
          let pt = point.position;

          pointsList.push([
            Math.round(width + pt.x),
            Math.round(height + pt.y),
            0
          ]);
        });

        this.fg_points.forEach(point => {
          let pt = point.position;

          pointsList.push([
            Math.round(width + pt.x),
            Math.round(height + pt.y),
            1
          ]);
        });

        axios
          .post(`/api/model/ritm/${this.$parent.image.id}`, {
            points: pointsList,
            ...this.settings
          })
          .then(response => {
            let segments = response.data.segmentaiton;
            let center = new paper.Point(width, height);
            let compoundPath = new paper.CompoundPath();
            for (let i = 0; i < segments.length; i++) {
              let polygon = segments[i];
              let path = new paper.Path();
              for (let j = 0; j < polygon.length; j += 2) {
                let point = new paper.Point(polygon[j], polygon[j + 1]);
                path.add(point.subtract(center));
              }
              path.closePath();
              compoundPath.addChild(path);
            }
            currentAnnotation.createCompoundPath();
            currentAnnotation.unite(compoundPath);
          })
          .finally(() => {
            // bg_points.forEach(point => point.remove());
            // fg_points.forEach(point => point.remove());
          });

        // this.clearPoints();
      }
    }
  }
};
</script>
