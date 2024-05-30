function getCenter(arr) {
    let centerLonLat = []
    if (arr.length) {
        const lon = []
        const lat = []
        const poly = [];
        for (let i = 0, len = arr.length; i < len; i++) {
            lon.push(arr[i][0])
            lat.push(arr[i][1])
        }
        for (let i = 0, len = lon.length; i < len; i++) {
            poly.push({
                x: parseFloat(lon[i]),
                y: parseFloat(lat[i]),
                z: 0
            });
        }
        const sortedLongitudeArray = poly.map(item => item.x).sort();
        const sortedLatitudeArray = poly.map(item => item.y).sort();
        const centerLongitude = ((parseFloat(sortedLongitudeArray[0]) + parseFloat(sortedLongitudeArray[sortedLongitudeArray.length - 1])) / 2).toFixed(14);
        const centerLatitude = ((parseFloat(sortedLatitudeArray[0]) + parseFloat(sortedLatitudeArray[sortedLatitudeArray.length - 1])) / 2).toFixed(14);
        console.log(centerLongitude, centerLatitude);
        centerLonLat = [Number(centerLongitude), Number(centerLatitude)]
    }
    return centerLonLat;
}
