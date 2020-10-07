
const React = require('react')
import { Link } from 'react-router'

export default React.createClass({
  render () {
    return (
      <div className='app'>
        <nav className='navbar navbar-inverse navbar-fixed-top'>
          <div className='container-fluid'>
            <div className='navbar-header'>
              <button type='button' className='navbar-toggle collapsed' data-toggle='collapse' data-target='#bs-example-navbar-collapse-1' aria-expanded='false'>
                <span className='sr-only'>Toggle navigation</span>
                <span className='icon-bar'></span>
                <span className='icon-bar'></span>
                <span className='icon-bar'></span>
              </button>
            </div>

            <div className='collapse navbar-collapse' id='bs-example-navbar-collapse-1'>
              <ul className='nav navbar-nav'>
                <li className='active explore'><Link to='/explore'>Explore</Link></li>
                <li className='compare'><Link to='/compare'>Compare</Link></li>
              </ul>
            </div>
          </div>
        </nav>

        <div className='app-view-container container-fluid'>
          {this.props.children}
        </div>
      </div>
    )
  }
})
