/* global $ */

const React = require('react')
const Exploration = require('./Exploration')

export default React.createClass({
  getInitialState () {
    let { query } = this.props.location
    return {
      params: {
        query: query ? query.query : '',
        limit: 1000,
        num_clusters: 30
      }
    }
  },
  componentDidMount () {
    $('.navbar-nav li').removeClass('active')
    $('.navbar-nav li.explore').addClass('active')
  },
  render () {
    return (
      <div className='row'>
        <div className='query-column col-md-2'>
          <div className='filters'>
            <form className='form' onSubmit={this._explore}>
              <div className='form-group'>
                <label htmlFor='queryInput'>Query:</label>
                <input id='queryInput' ref='queryInput' className='form-control' type='text' defaultValue={this.state.params.query}></input>
              </div>
              <div className='form-group'>
                <label htmlFor='limitInput'>Num Vectors:</label>
                <input id='limitInput' ref='limitInput' className='form-control' type='text' defaultValue={this.state.params.limit}></input>
              </div>
              <div className='form-group'>
                <label htmlFor='limitInput'>Num Clusters:</label>
                <input id='numClustersInput' ref='numClustersInput' className='form-control' type='text' defaultValue={this.state.params.num_clusters}></input>
              </div>
              <div className='form-group'>
                <input type='submit' className='btn btn-primary' value='Explore'/>
              </div>
            </form>
          </div>
        </div>
        <Exploration ref='exploration' filter={this.state.params} onDrillOut={this._onDrillOut} />
      </div>
    )
  },
  _explore (e) {
    e && e.preventDefault()
    var query = this.refs.queryInput.value
    var limit = parseInt(this.refs.limitInput.value, 10)
    var numClusters = parseInt(this.refs.numClustersInput.value, 10)
    var params = this.state.params
    params.query = query
    params.limit = limit
    params.num_clusters = numClusters
    this.props.history.pushState(null, '/explore', {query: query})
    this.setState({params})
    this.refs.exploration.setState({params})
    this.refs.exploration.explore()
  },
  _onDrillOut (params) {
    this.refs.queryInput.value = params.query
    this._explore()
  }
})
