
const React = require('react')

export default React.createClass({
  render () {
    let data = this.props.data
    const format = (x) => x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',')
    const numVectors = format(data.num_vectors)
    const vocabSize = format(data.vocab_size)
    let sampleRate = null
    if (data.sample_rate) sampleRate = (Math.round(data.sample_rate * 100000) / 100000)
    return (
      <div className='stats'>
        Showing {numVectors} out of {vocabSize} vectors. {sampleRate !== null && (<span>Sample Rate = {sampleRate}</span>)}
      </div>
    )
  }
})
