
const React = require('react')
const Api = require('./../utils/Api')
const ScatterPlot2d = require('./ScatterPlot2d')
const VectorList = require('./VectorList')
const ClusterList = require('./ClusterList')
const Stats = require('./Stats')
const d3 = require('d3')

export default React.createClass({
  getInitialState () {
    return {
      params: this.props.filter,
      loading: false,
      result: null,
      error: null,
      color: d3.scale.category20c()
    }
  },
  componentWillMount () {
    this.explore()
  },
  componentWillUpdate () {
  },
  render () {
    let result = this.state.result
    let vectorListTitle = 'Most Similar'
    let params = this.state.params
    if (params && !params.query) vectorListTitle = 'Sample Rated'
    return (
      <div className='exploration'>
        {(this.state.error) && (
          <div className='alert alert-danger'>{this.state.error.message}</div>
        )}
        {(this.state.loading) && (
          <div className='loader'><div className='spinner'></div></div>
        )}
        {(result) && (
          <div className='result'>
            <div className='col-md-8 center-pane'>
              {(result) && (result.stats) && (
                <Stats data={result.stats} />
              )}
              <ScatterPlot2d ref='plot' color={this.state.color} points={result.reduction} clusters={result.clusters} labels={result.labels} />
            </div>
            <div className='col-md-2 right-pane'>
              <div className='split-pane upper'>
                <VectorList ref='mostSimilarList' title={vectorListTitle} data={result} onSelect={this._onDrillDown} />
              </div>
              <div className='split-pane lower'>
                <ClusterList ref='clusterList' color={this.state.color} title='K-Means Centroids' data={result} />
              </div>
            </div>
          </div>
        )}
      </div>
    )
  },
  explore () {
    let params = this.state.params
    this.setState({loading: true, error: null})
    Api.request('GET', '/explore', {
      query: params.query,
      limit: (params.limit || 1000),
      enable_clustering: true,
      num_clusters: params.num_clusters
    }, (error, result) => {
      if (result) {
        this.refs.plot && this.refs.plot.setState({
          points: result.reduction,
          clusters: result.clusters,
          labels: result.labels
        })
        this.refs.mostSimilarList && this.refs.mostSimilarList.setState({selected: null})
      }
      let loading = false
      this.setState({error, result, loading})
    })
  },
  _onDrillDown (params) {
    this.props.onDrillOut(params)
  }
})
