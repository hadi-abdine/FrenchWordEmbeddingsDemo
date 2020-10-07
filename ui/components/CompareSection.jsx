/* global $ */

const React = require('react')
const Api = require('./../utils/Api')
const ScatterPlot2d = require('./ScatterPlot2d')

export default React.createClass({
  getInitialState () {
    return {
      error: null,
      loading: false,
      data: null,
      queryA: '',
      queryB: ''
    }
  },
  componentDidMount () {
    $('.navbar-nav li').removeClass('active')
    $('.navbar-nav li.compare').addClass('active')
  },
  render () {
    const result = this.state.result
    const queryA = this.state.queryA
    const queryB = this.state.queryB
    const axesLabels = this._axesLabels(queryA, queryB)
    return (
      <div className='row'>
        <div className='query-column col-md-3'>
          <div className='filters'>
            <form className='form' onSubmit={this._onCompareSubmit}>
              <div className='form-group'>
                <label htmlFor='queryAInput'>Query A:</label>
                <input id='queryAInput' ref='queryAInput' className='form-control' type='text' defaultValue={queryA} />
              </div>
              <div className='form-group'>
                <label htmlFor='queryBInput'>Query B:</label>
                <input id='queryBInput' ref='queryBInput' className='form-control' type='text' defaultValue={queryB} />
              </div>
              <div className='form-group'>
                <input type='submit' className='btn btn-primary' value='Compare'/>
              </div>
            </form>
          </div>
        </div>
        <div className='result comparison'>
          <div className='center-pane col-md-9'>
            {result && (
              <ScatterPlot2d ref='plot' labels={result.labels} points={result.comparison} axes={axesLabels} showLabels='true' />
            )}
          </div>
        </div>
      </div>
    )
  },
  compare (queryA, queryB) {
    this.setState({loading: true, error: null, result: null})
    Api.request('GET', '/compare', {
      queries: [queryA, queryB],
      limit: 30
    }, (error, result) => {
      let loading = false
      this.setState({error, result, loading})
    })
  },
  _onCompareSubmit (e) {
    e && e.preventDefault()
    var queryA = this.refs.queryAInput.value
    var queryB = this.refs.queryBInput.value
    this.setState({queryA, queryB})
    this.compare(queryA, queryB)
  },
  _axesLabels (queryA, queryB) {
    return [queryA.split(' AND ')[0], queryB.split(' AND ')[0]]
  }
})
