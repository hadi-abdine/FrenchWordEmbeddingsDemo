/* global $, localStorage */

const React = require('react')
const d3 = require('d3')

export default React.createClass({
  getInitialState () {
    return {
      points: this.props.points,
      clusters: this.props.clusters,
      labels: this.props.labels,
      viewOptions: {
        showLabels: this.props.showLabels || localStorage.scatter2dShowLabels
      }
    }
  },
  componentDidMount () {
    this._renderD3()
  },
  componentWillUpdate () {
    this._updateD3()
  },
  render () {
    let viewOptions = this.state.viewOptions
    let scatterClasses = 'scatter-plot-2d'
    if (viewOptions.showLabels) scatterClasses += ' show-labels'
    this._updateD3 && this._updateD3()
    return (
      <div className='scatter-plot-2d-container'>
        <div className='view-options'>
          <div className='checkbox'>
            <label>
              <input
                ref='showLabelsInput'
                type='checkbox'
                defaultValue='yes'
                checked={viewOptions.showLabels}
                onChange={this._setViewOptions} /> Show Labels
            </label>
          </div>
        </div>
        <div id='scatter-plot-2d' ref='scatterPlot2dElement' className={scatterClasses}>
        </div>
      </div>
    )
  },
  _setViewOptions () {
    var viewOptions = {
      showLabels: this.refs.showLabelsInput.checked
    }
    if (viewOptions.showLabels) {
      localStorage.scatter2dShowLabels = 'true'
    } else {
      delete localStorage.scatter2dShowLabels
    }
    this.setState({viewOptions: viewOptions})
  },
  _nodeRadius () {
    var nodeRadius = 2
    if (this.state.points.length > 5000) nodeRadius = 1
    if (this.state.points.length < 500) nodeRadius = 3
    if (this.state.points.length < 50) nodeRadius = 4
    return nodeRadius
  },
  _maxLabelTextSize () {
    return 20
  },
  _nominalNodeLabelTextSize () {
    var size = 10
    if (this.state.points.length >= 10000) size = 1
    if (this.state.points.length >= 1000) size = 2
    if (this.state.points.length >= 500) size = 5
    if (this.state.points.length <= 500) size = 15
    if (this.state.points.length <= 50) size = 20
    return size
  },
  _renderD3 () {
    var dataset = this.state.points
    var self = this

    var colors = this.props.color || d3.scale.category20c()
    var $scatterPlot2dElement = $(this.refs.scatterPlot2dElement)
    var width = $scatterPlot2dElement.width()
    var height = $scatterPlot2dElement.height()
    var padding = 20
    var labelNodes = null

    console.log(`ScatterPlot2d width=${width}, height=${height}, dataset.length=${dataset.length}`)

    var zoom = d3.behavior.zoom()
      .scaleExtent([1, 10])
      .on('zoom', zoomed)

    d3.behavior.drag()
      .origin((d) => d)
      .on('dragstart', dragstarted)
      .on('drag', dragged)
      .on('dragend', dragended)

    function zoomed () {
      container.attr('transform', 'translate(' + d3.event.translate + ')scale(' + d3.event.scale + ')')
      if (labelNodes) {
        var labelTextSize = self._nominalNodeLabelTextSize()
        if (self._nominalNodeLabelTextSize() * zoom.scale() > self._maxLabelTextSize()) labelTextSize = self._maxLabelTextSize() / zoom.scale()
        console.log('Setting labelTextSize (onZoom)', labelTextSize)
        labelNodes.style('font-size', labelTextSize + 'px')
      }
    }

    function dragstarted (d) {
      d3.event.sourceEvent.stopPropagation()
      d3.select(this).classed('dragging', true)
    }

    function dragged (d) {
      d3.select(this).attr('cx', d.x = d3.event.x).attr('cy', d.y = d3.event.y)
    }

    function dragended (d) {
      d3.select(this).classed('dragging', false)
    }

    var xScale = d3.scale.linear()
      .domain([d3.min(dataset, (d) => d[0]), d3.max(dataset, (d) => d[0])])
      .range([padding, width - padding])

    var yScale = d3.scale.linear()
      .domain([d3.min(dataset, (d) => d[1]), d3.max(dataset, (d) => d[1])])
      .range([height - padding, padding])

    var xAxis = d3.svg.axis()
      .scale(xScale)
      .orient('bottom')
      .ticks(5)

    var yAxis = d3.svg.axis()
      .scale(yScale)
      .orient('left')
      .ticks(5)

    var svg = d3.select(this.refs.scatterPlot2dElement)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', 'translate(' + 0 + ',' + 0 + ')')
      .call(zoom)

    svg.append('rect')
      .attr('width', width)
      .attr('height', height)
      .style('fill', 'white')
      .style('pointer-events', 'all')

    var container = svg.append('g')

    if (this.props && this.props.axes && this.props.axes.length) {
      console.log('Drawing axes labels')

      container.append('g')
        .attr('class', 'x axis')
        .attr('transform', 'translate(0,' + (height - padding) + ')')
        .call(xAxis)
        .append('text')
        .attr('class', 'axis-label')
        .attr('x', width - padding)
        .attr('y', -6)
        .style('text-anchor', 'end')
        .text(this.props.axes[0])

      container.append('g')
        .attr('class', 'y axis')
        .attr('transform', 'translate(' + padding + ',0)')
        .call(yAxis)
        .append('text')
        .attr('class', 'axis-label')
        .attr('transform', 'rotate(-90)')
        .attr('y', 6)
        .attr('x', 0 - padding)
        .attr('dy', '.71em')
        .style('text-anchor', 'end')
        .text(this.props.axes[1])
    }

    function update () {
      let dataset = this.state.points
      let clusters = this.state.clusters
      let labels = this.state.labels
      let viewOptions = this.state.viewOptions

      console.log(`ScatterPlot2d update, dataset.length=${dataset.length}`)

      xScale.domain([d3.min(dataset, (d) => d[0]), d3.max(dataset, (d) => d[0])])
      yScale.domain([d3.min(dataset, (d) => d[1]), d3.max(dataset, (d) => d[1])])

      // Update data for existing nodes
      var nodes = container.selectAll('circle.node')
        .data(dataset)

      // Change existing nodes
      nodes
        .transition()
        .duration(1000)
        .delay((d, i) => i / dataset.length * 500)
        .attr('cx', (d) => xScale(d[0]))
        .attr('cy', (d) => yScale(d[1]))
        .attr('fill', (d, i) => colors(clusters ? clusters[i] : 0))
        .attr('r', this._nodeRadius())

      // Render new nodes
      nodes
        .enter()
        .append('circle')
        .transition()
        .duration(1000)
        .delay((d, i) => i / dataset.length * 500)
        .each('start', function () {
          d3.select(this)
            .attr('fill', 'white')
        })
        .each('end', function (d, i) {
          d3.select(this)
            .attr('fill', () => colors(clusters ? clusters[i] : 0))
        })
        .attr('class', 'node')
        .attr('cx', (d) => xScale(d[0]))
        .attr('cy', (d) => yScale(d[1]))
        .attr('fill', (d, i) => colors(clusters ? clusters[i] : 0))
        .attr('r', this._nodeRadius())

      if (viewOptions.showLabels) {
        labelNodes = container.selectAll('text.node-label')
          .data(dataset)

        var labelTextSize = this._nominalNodeLabelTextSize()
        if (this._nominalNodeLabelTextSize() * zoom.scale() > this._maxLabelTextSize()) labelTextSize = this._maxLabelTextSize() / zoom.scale()
        console.log('Setting labelTextSize', labelTextSize)
        labelNodes.style('font-size', labelTextSize + 'px')

        labelNodes
          .transition()
          .duration(1000)
          .delay((d, i) => i / dataset.length * 500)
          .attr('x', (d) => xScale(d[0]))
          .attr('y', (d) => yScale(d[1]))

        labelNodes
          .enter()
          .append('text')
          .attr('class', 'node-label')
          .attr('text-anchor', 'middle')
          .attr('x', (d) => xScale(d[0]))
          .attr('y', (d) => yScale(d[1]))
          .attr('dy', 0 - this._nodeRadius() - 1.5)
          .style('font-size', labelTextSize + 'px')
          .text((d, i) => labels[i])

        labelNodes
          .exit()
          .remove()
      } else {
        container.selectAll('text.node-label')
          .remove()
      }

      // Remove old nodes
      nodes
        .exit()
        .remove()

      // Update X Axis
      container.select('.x.axis')
        .transition()
        .duration(1000)
        .call(xAxis)

      // Update Y Axis
      container.select('.y.axis')
        .transition()
        .duration(100)
        .call(yAxis)
    }

    this._updateD3 = update.bind(this)
    this._updateD3()
  }
})
