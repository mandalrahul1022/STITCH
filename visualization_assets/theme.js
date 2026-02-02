const ThemeBoot = function () {
  if (!document.getElementById("stellar-backdrop")) {
    const wrapper = document.createElement("div");
    wrapper.id = "stellar-backdrop";
    wrapper.innerHTML =
      '<div class="stars"></div><div class="twinkles"></div>';
    document.body.appendChild(wrapper);
  }

  this.render = function () {
    /* no-op: theme does not consume model data */
  };

  this.reset = function () {
    /* nothing to reset */
  };
};
