/* global $ */

const React = require('react')

export default React.createClass({
  getInitialState () {
    return {
      selected: null
    }
  },
  render () {
    let data = this.props.data
    let color = this.props.color
    let itemsHtml = []
    for (let i = 0; data.cluster_centroids_closest_nodes.length > i; i++) {
      let className = 'list-group-item'
      if (this.state.selected === i) className += ' active'
      let closestNodeId = data.cluster_centroids_closest_nodes[i]
      let label = data.labels[closestNodeId]
      let numClusters = data.clusters.filter((cluster) => cluster === i).length
      let badgeStyle = {backgroundColor: color(i)}
      itemsHtml.push(<a className={className} key={i} onClick={this._onClick} data-index={i}><span className='badge' style={badgeStyle}>{numClusters}</span>{label}</a>)
    }
    return (
      <div className='cluster-list panel panel-default'>
        <div className='panel-heading'>{this.props.title}</div>
        <div className='list-group'>
          {itemsHtml}
        </div>
      </div>
    )
  },
  _onClick (e) {
    var index = parseInt($(e.target).attr('data-index'), 10)
    this.setState({selected: index})
  }
})
