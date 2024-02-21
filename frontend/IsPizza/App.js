import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, Image, Dimensions, Pressable } from 'react-native';


const img = require('./assets/Pizza.jpg');
const {width, height} = Dimensions.get('window');
const textColor = '#fff';

export default function App() {
  return (
    <View style={styles.container}>
      <View style={styles.imageContainer}>
        <Image source={img} style={styles.image} />
      </View>
      <View style={styles.outputContainer}>
        <OutputText output = "Someoutput"/>
      </View>
      <View style={styles.buttonsContainer}>
        <Button label='Choose a picture from Photos'/>
        <Button label='Take a new Photo'/>
      </View>
      <StatusBar style='auto' />
    </View>
  );
}

function Button({ label }) {
  return (
    <View style={styles.buttonContainer}>
      <Pressable style={styles.button} onPress={() => alert('You pressed a button.')}>
        <Text style={styles.buttonLabel}>{label}</Text>
      </Pressable>
    </View>
  );
}
function OutputText({ result }){
  return (
    <View style={styles.buttonContainer}>
      <Text>result</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    height: height,
    width: width,
    paddingTop: height* 0.1,
    paddingBottom: height* 0.1,
    flex: 1,
    backgroundColor: '#2E4053',
    alignItems: 'center',
    justifyContent: 'center',
  },
  imageContainer: {
    flex: 2/5,
  },
  image: {
    // flex: 1/3,
    width: width * 0.9,
    height: height * 0.4,
    borderRadius: 18,
  },
  buttonsContainer: {
    flex: 2/5,
    alignItems: 'center',
  },
  buttonContainer: {
    width: '80%',
    height: '10%',
    marginHorizontal: 20,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 3,
  },
  button: {
    borderRadius: 10,
    width: '100%',
    height: '100%',
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
  },
  buttonIcon: {
    paddingRight: 8,
  },
  buttonLabel: {
    color: textColor,
    fontSize: 16,
  },
  outputContainer: {
    flex: 1/5,
    alignItems: 'center',
    color: textColor,
    fontSize: 16,
  },
  output: {
    flex: 1/5,
  }
});
