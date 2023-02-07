<template>
  <div v-show="ritm.isActive">
    <PanelInputNumber
      name="Padding"
      min="0"
      max="1000"
      step="2"
      v-model="ritm.settings.padding"
    />
    <PanelInputNumber
      name="Threshold"
      min="0"
      max="100"
      step="5"
      v-model="ritm.settings.threshold"
    />
  </div>
</template>

<script>
import PanelInputNumber from "@/components/PanelInputNumber";

export default {
  name: "RITMPanel",
  components: { PanelInputNumber },
  props: {
    ritm: {
      type: Object,
      required: true
    }
  }
};
</script>
